from utils import *
from src import *
from functools import partial


def get_trunc_node_pos_fn(mov_node_size, data):
    node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4 
    node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4
    def trunc_node_pos_fn(x):
        if torch.isnan(x).any():
            raise ValueError("node position contains NaN values.")
        x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
        return x
    return trunc_node_pos_fn

def generate_log_string(ps, iteration, hpwl, overflow, obj=None):
    base_str = "iter: %d | masked_hpwl: %.2E overflow: %.4f density_weight: %.4E" % (iteration, hpwl, overflow, ps.density_weight)
    if ps.open_route_force_opt:
        return base_str + " w : d : all : r : cg : p = %.1f %.3f %.3f %.3f %.3f %.3f" % (
            1.0, ps.density_force_ratio, ps.all_route_force_ratio, ps.mov_route_force_ratio, ps.mov_congest_force_ratio, ps.mov_pseudo_force_ratio
        )
    else:
        return base_str + " obj: %.4E wa_coeff: %.4E" % (obj, ps.wa_coeff)
    
def run_placement_main_nesterov(args, logger):
    total_start = time.time()
    data, rawdb, gpdb = load_dataset(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    assert args.use_eplace_nesterov
    logger.info("Use Nesterov optimizer!")
    if args.scale_design:
        logger.warning("Eplace's nesterov optimizer cannot support normalized die. Disable scale_design.")
        args.scale_design = False
    data = data.to(device)
    data = data.preprocess()
    logger.info(data)
    logger.info(data.node_type_indices)
    # args.num_bin_x = args.num_bin_y = 2 ** math.ceil(math.log2(max(data.die_info).item() // 25))

    init_density_map = get_init_density_map(rawdb, gpdb, data, args, logger)
    data.init_filler()
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info()
    mov_node_pos = mov_node_pos.requires_grad_(True)

    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)

    conn_fix_node_pos = data.node_pos.new_empty(0, 2)
    if data.fixed_connected_index[0] < data.fixed_connected_index[1]:
        lhs, rhs = data.fixed_connected_index
        conn_fix_node_pos = data.node_pos[lhs:rhs, ...]
    conn_fix_node_pos = conn_fix_node_pos.detach()

    def overflow_fn(mov_density_map):
        overflow_sum = ((mov_density_map - args.target_density) * data.bin_area).clamp_(min=0.0).sum()
        return overflow_sum / data.total_mov_area_without_filler
    overflow_helper = (mov_lhs, mov_rhs, overflow_fn)

    ps = ParamScheduler(data, args, logger)
    density_map_layer = ElectronicDensityLayer(
        unit_len=data.unit_len,
        num_bin_x=data.num_bin_x,
        num_bin_y=data.num_bin_y,
        device=device,
        overflow_helper=overflow_helper,
        sorted_maps=data.sorted_maps,
        expand_ratio=expand_ratio,
        deterministic=args.deterministic,
    ).to(device)

    # fix_lhs, fix_rhs = data.fixed_index
    # info = (0, 0, data.design_name + "_fix")
    # fix_node_pos = data.node_pos[fix_lhs:fix_rhs, ...]
    # fix_node_size = data.node_size[fix_lhs:fix_rhs, ...]
    # draw_fig_with_cairo(
    #     None, None, fix_node_pos, fix_node_size, None, None, data, info, args
    # )
    evaluator_fn = partial(
        fast_evaluator,
        constraint_fn=trunc_node_pos_fn,
        mov_node_size=mov_node_size,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
    )
    
    def calc_route_force(mov_node_pos, mov_node_size, expand_ratio, constraint_fn, evaluator_fn=None):
        return get_route_force(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
            constraint_fn=constraint_fn, evaluator_fn=evaluator_fn
        )

    calc_route_force = partial(
        calc_route_force,
        evaluator_fn=evaluator_fn
    )        
    
    obj_and_grad_fn = partial(
        calc_obj_and_grad,
        constraint_fn=trunc_node_pos_fn,
        route_fn=calc_route_force,
        mov_node_size=mov_node_size,
        expand_ratio=expand_ratio,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
    )
    
    def initialize_optimizer(obj_and_grad_fn, mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
                             density_map_layer, mov_node_size, expand_ratio, init_density_map, 
                             ps, data, args, route_fn):
        optimizer = NesterovOptimizer([mov_node_pos], lr=0)

        # initialization
        init_params(
            mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
            density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer, 
            ps, data, args, route_fn=route_fn
        )

        # init learning rate
        lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr.item()
        logger.info("Reset optimizer...  lr: %.2E " % (lr.item()))
            
        return optimizer, lr
    
    optimizer,_ = initialize_optimizer(obj_and_grad_fn, mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, 
                                    conn_fix_node_pos, density_map_layer, mov_node_size, expand_ratio, 
                                    init_density_map, ps, data, args, calc_route_force)

    torch.cuda.synchronize(device)
    gp_start_time = time.time()
    logger.info("start gp")

    route_force_init_times = 0
    terminate_signal = False
    route_early_terminate_signal = False
    for iteration in range(args.inner_iter):
        # optimizer.zero_grad() # zero grad inside obj_and_grad_fn
    
        obj = optimizer.step(obj_and_grad_fn)
        hpwl, overflow = evaluator_fn(mov_node_pos)
        # update parameters
        ps.step(hpwl, overflow, mov_node_pos, data)
        if ps.need_to_early_stop():
            terminate_signal = True

        # for only cell inflation
        if ps.use_cell_inflate_only and terminate_signal and ps.curr_optimizer_cnt < ps.max_route_opt:
            terminate_signal = False  # reset signal
            ps.open_cell_inflate_opt = True # open cell inflation
            ps.curr_optimizer_cnt += 1
            best_res = ps.get_best_solution()
            if best_res[0] is not None:
                best_sol, hpwl, overflow = best_res
                mov_node_pos.data.copy_(best_sol)

        # for route force
        if ps.use_route_force and terminate_signal:
            best_res = ps.get_best_solution()
            ps.reset_best_sol()
            if best_res[0] is not None:
                best_sol, hpwl, overflow = best_res
                mov_node_pos.data.copy_(best_sol)
                ps.push_wait_router_sol(hpwl, overflow, best_sol)
                            
            if ps.curr_optimizer_cnt < ps.max_route_force_opt:
                terminate_signal = False  # reset signal
                ps.curr_optimizer_cnt += 1
                # open route force optimization
                ps.start_route_iter = iteration
                ps.open_route_force_opt, ps.rerun_route, ps.recal_conn_route_force = True, True, True
                logger.info("iter: %d |Open route force optimization..." % iteration)
            
        # rerun route 
        if ps.open_route_force_opt and ps.need_to_stop_route_force_opt():
            ps.open_route_force_opt = False
            best_sol_gr = ps.get_best_gr_sol()
            mov_node_pos.data.copy_(best_sol_gr)
            ps.reset_gr_sol_recorder()
            ps.use_norm_density_weight = True
            optimizer, _ = initialize_optimizer(obj_and_grad_fn, mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, 
                                                conn_fix_node_pos, density_map_layer, mov_node_size, expand_ratio, 
                                                init_density_map, ps, data, args, calc_route_force)
            logger.info("Iter: %d | density_weight: %.2E route_weight: %.2E congest_weight: %.2E pseudo_weight: %.2E " 
                % (iteration,ps.density_weight,ps.route_weight,ps.congest_weight,ps.pseudo_weight,))
        
        if ps.open_route_force_opt and (iteration % args.route_freq == 0):
            ps.rerun_route, ps.recal_conn_route_force= True, True
                    
        if ps.open_route_force_opt and ps.rerun_route:
            new_mov_node_size, new_expand_ratio = None, None
            if ps.use_cell_inflate and route_force_init_times < 1000:
                output = route_inflation(
                    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
                    constraint_fn=trunc_node_pos_fn, rerun_route=ps.rerun_route
                )  # ps.use_cell_inflate is updated in route_inflation
                if not ps.use_cell_inflate:
                    logger.info("Early stop cell inflation...")
                if output is not None:
                    gr_metrics, new_mov_node_size, new_expand_ratio = output
                if ps.use_cell_inflate and new_mov_node_size is not None:
                    # remove some fillers, we should update the size the pos
                    mov_node_size = new_mov_node_size
                    mov_node_pos = mov_node_pos[:new_mov_node_size.shape[0]].detach().clone()
                    mov_node_pos = mov_node_pos.requires_grad_(True)
                    # update expand ratio and precondition relevant data
                    expand_ratio = new_expand_ratio  # already update in route_inflation()
                    data.mov_node_to_num_pins = data.mov_node_to_num_pins[:new_mov_node_size.shape[0]]
                    data.mov_node_area = data.mov_node_area  # already update in route_inflation()
                    # update partial function correspondingly
                    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)
                    obj_and_grad_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    obj_and_grad_fn.keywords["mov_node_size"] = mov_node_size
                    obj_and_grad_fn.keywords["expand_ratio"] = expand_ratio
                    evaluator_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    evaluator_fn.keywords["mov_node_size"] = mov_node_size
                    density_map_layer.expand_ratio = expand_ratio

            if route_force_init_times<100000:
                route_force_init_times += 1
                optimizer, _ = initialize_optimizer(obj_and_grad_fn, mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, 
                                                conn_fix_node_pos, density_map_layer, mov_node_size, expand_ratio, 
                                                init_density_map, ps, data, args, calc_route_force)
                logger.info("Iter: %d | density_weight: %.2E route_weight: %.2E congest_weight: %.2E pseudo_weight: %.2E " 
                    % (iteration,ps.density_weight,ps.route_weight,ps.congest_weight,ps.pseudo_weight,))
            
        if ps.open_cell_inflate_opt and ps.use_cell_inflate_only:
            ps.open_cell_inflate_opt = False
            ps.prev_optimizer_cnt = ps.curr_optimizer_cnt
            
            new_mov_node_size, new_expand_ratio = None, None
            output = route_inflation(
                args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
                constraint_fn=trunc_node_pos_fn, rerun_route=True
            )  # ps.use_cell_inflate is updated in route_inflation
            if not ps.use_cell_inflate:
                route_early_terminate_signal = True
                terminate_signal = True
                logger.info("Early stop cell inflation...")
            if output is not None:
                gr_metrics, new_mov_node_size, new_expand_ratio = output
                ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
            if ps.use_cell_inflate:
                if new_mov_node_size is not None:
                    # remove some fillers, we should update the size the pos
                    mov_node_size = new_mov_node_size
                    mov_node_pos = mov_node_pos[:new_mov_node_size.shape[0]].detach().clone()
                    mov_node_pos = mov_node_pos.requires_grad_(True)
                    # update expand ratio and precondition relevant data
                    expand_ratio = new_expand_ratio  # already update in route_inflation()
                    data.mov_node_to_num_pins = data.mov_node_to_num_pins[:new_mov_node_size.shape[0]]
                    data.mov_node_area = data.mov_node_area  # already update in route_inflation()
                    # update partial function correspondingly
                    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)
                    obj_and_grad_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    obj_and_grad_fn.keywords["mov_node_size"] = mov_node_size
                    obj_and_grad_fn.keywords["expand_ratio"] = expand_ratio
                    evaluator_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    evaluator_fn.keywords["mov_node_size"] = mov_node_size
                    density_map_layer.expand_ratio = expand_ratio
                    
                # reset nesterov optimizer
                optimizer,cur_lr = initialize_optimizer(obj_and_grad_fn, mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, 
                                                conn_fix_node_pos, density_map_layer, mov_node_size, expand_ratio, 
                                                init_density_map, ps, data, args, calc_route_force)
                logger.info(
                    "Route Iter: %d | lr: %.2E density_weight: %.2E route_weight: %.2E "
                    "congest_weight: %.2E pseudo_weight: %.2E " 
                    % (
                        ps.curr_optimizer_cnt - 1,
                        cur_lr.item(),
                        ps.density_weight,
                        ps.route_weight,
                        ps.congest_weight,
                        ps.pseudo_weight,
                    )
                )
                ps.reset_best_sol()
                
        if iteration % args.log_freq == 0 or iteration == args.inner_iter - 1 or ps.rerun_route or terminate_signal:
            log_str = generate_log_string(ps, iteration, hpwl, overflow, obj)
            logger.info(log_str)

            #if args.draw_placement and ps.open_route_force_opt and iteration % 1 == 0:
            if args.draw_placement and iteration % 1000 == 0 and ps.open_route_force_opt:
                info = (iteration, hpwl, data.design_name)

                mov_node_pos_to_draw = mov_node_pos[mov_lhs:mov_rhs, ...].clone()
                mov_node_size_to_draw = data.node_size[mov_lhs:mov_rhs, ...].clone()
                fix_node_pos_to_draw = data.node_pos[mov_rhs:, ...].clone()
                fix_node_size_to_draw = data.node_size[mov_rhs:, ...].clone()

                #node_pos_to_draw = torch.cat([mov_node_pos_to_draw, fix_node_pos_to_draw], dim=0)
                #node_size_to_draw = torch.cat([mov_node_size_to_draw, fix_node_size_to_draw], dim=0)
                node_pos_to_draw = mov_node_pos_to_draw
                node_size_to_draw = mov_node_size_to_draw

                filler_node_pos_to_draw, filler_node_size_to_draw = None, None
                if args.use_filler:
                    filler_node_pos_to_draw = mov_node_pos[mov_rhs:, ...].clone()
                    filler_node_size_to_draw = data.filler_size[:(mov_node_pos.shape[0] - mov_rhs), ...]
                    node_pos_to_draw = torch.cat([node_pos_to_draw, filler_node_pos_to_draw], dim=0)
                    node_size_to_draw = torch.cat([node_size_to_draw, filler_node_size_to_draw], dim=0)
                    

                # draw_fig_with_cairo_cpp(node_pos_to_draw, node_size_to_draw, data, info, args)
                
                draw_fig_with_cairo(
                    mov_node_pos=mov_node_pos_to_draw,
                    mov_node_size=mov_node_size_to_draw,
                    node_pos = node_pos_to_draw,
                    node_size = node_size_to_draw,
                    fix_node_pos=fix_node_pos_to_draw,
                    fix_node_size=fix_node_size_to_draw,
                    filler_node_pos=filler_node_pos_to_draw,
                    filler_node_size=filler_node_size_to_draw,
                    data=data,
                    info=info,
                    route_grad=ps.grad_recorder["all_route_grad"],
                    args=args
                )
                
        if terminate_signal:
            break

    # Save best solution
    best_res = ps.get_best_solution()
    if best_res[0] is not None:
        best_sol, hpwl, overflow = best_res
        # fillers are unused from now, we don't copy there data
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol[mov_lhs:mov_rhs])
    if ps.enable_route:
        route_inflation_roll_back(args, logger, data, mov_node_size)
        if ps.use_cell_inflate_only:
            if not route_early_terminate_signal:
                gr_metrics = run_gr_and_fft_main(
                args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=trunc_node_pos_fn, 
                skip_m1_route=True, report_gr_metrics_only=True, rerun_route=True
                )
                ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
        elif ps.use_route_force:
            for wait_router_sol in ps.wait_router_sol_recorder:
                hpwl, overflow, mov_node_pos = wait_router_sol[0], wait_router_sol[1], wait_router_sol[2]
                gr_metrics = run_gr_and_fft_main(
                    args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=trunc_node_pos_fn, 
                    skip_m1_route=True, report_gr_metrics_only=True, rerun_route=True
                )
                ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
            ps.reset_wait_router_sol_recorder()

        best_sol_gr = ps.get_best_gr_sol()
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol_gr[mov_lhs:mov_rhs])
        ps.reset_gr_sol_recorder()

    node_pos = mov_node_pos[mov_lhs:mov_rhs]
    node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
    torch.cuda.synchronize(device)
    gp_end_time = time.time()
    gp_time = gp_end_time - gp_start_time
    gp_per_iter = gp_time / (iteration + 1)
    logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f GP Time: %.4fs perIterTime: %.6fs" % 
        (iteration, hpwl, overflow, gp_time, gp_time / (iteration + 1))
    )

    # Eval
    hpwl, overflow = evaluate_placement(
        node_pos, density_map_layer, init_density_map, data, args
    )
    hpwl, overflow = hpwl.item(), overflow.item()
    info = (iteration + 1, hpwl, data.design_name)
    if args.draw_placement:
        draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
    logger.info("After GP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f" % (hpwl, overflow))
    ps.visualize(args, logger)
    gp_hpwl = hpwl
    gp_time = gp_end_time - gp_start_time
    iteration += 1 # increase 1 For DP drawing

    # detail placement
    node_pos, dp_hpwl, top5overflow, lg_time, dp_time = detail_placement_main(
        node_pos, gpdb, rawdb, ps, data, args, logger
    )
    iteration += 1
    ps.iter += 1
    #if args.generate_gif:
    #    generate_gif(args, data.design_name)
        
    route_metrics = None
    if args.final_route_eval:
        logger.info("Final routing evalution by GGR...")
        route_metrics = run_gr_and_fft(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos[mov_lhs:mov_rhs],
            report_gr_metrics_only=True,
            skip_m1_route=True, given_gr_params={
                "rrrIters": 1,
                "route_guide": os.path.join(args.result_dir, args.exp_id, args.output_dir, "%s_%s.guide" %(args.output_prefix, args.design_name)),
            }
        )

    if args.load_from_raw:
        del gpdb, rawdb

    place_time = time.time() - total_start
    logger.info("GP Time: %.4f LG Time: %.4f DP Time: %.4f Total Place Time: %.4f" % (
        gp_time, lg_time, dp_time, place_time))
    place_metrics = (dp_hpwl, gp_hpwl, top5overflow, overflow, gp_time, dp_time + lg_time, gp_per_iter, place_time)

    return place_metrics, route_metrics