import torch
from cpp_to_py import gpugr
from .dct2_fft2 import dct2, idct2, idxst_idct, idct_idxst
from .torch_dct import torch_dct_idct
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import copy
import torchvision.transforms
from utils import *


class RouteCache:
    def __init__(self) -> None:
        self.first_run = True
        self.first_cell_inflate_run = True
        self.grdb = None
        self.gr_and_fft_main_output = None
        self.route_input_mat: torch.Tensor = None
        self.routeforce = None
        self.route_gradmat: torch.Tensor = None
        self.route_gradmat_upset: torch.Tensor = None
        self.cg_mapAll: torch.Tensor = None
        self.mov_route_grad: torch.Tensor = None
        self.placeable_area = None
        self.target_area = None
        self.whitespace_area = None
        self.mov_node_size_real: torch.Tensor = None
        self.original_filler_area_total = None
        self.original_pin_rel_cpos: torch.Tensor = None
        self.original_target_density = None
        self.original_total_mov_area_without_filler = None
        self.original_num_fillers = None
        self.classfied_cell_indices = {}
        self.original_mov_node_size: torch.Tensor = None
        self.original_mov_node_size_real: torch.Tensor = None
        self.original_init_density_map: torch.Tensor = None
        # for momentum cell inflation
        self.inflate_ratio_prev: torch.Tensor = None
        self.delta_inflate_ratio_prev: torch.Tensor = None
        self.inflate_ratio_from_cg_map_recorder = []

    def reset(self):
        self.first_run = True
        self.first_cell_inflate_run = True
        self.grdb = None
        self.gr_and_fft_main_output = None
        self.route_input_mat = None
        self.routeforce = None
        self.route_gradmat = None
        self.route_gradmat_upset = None
        self.cg_mapAll = None
        self.mov_route_grad = None
        self.placeable_area = None
        self.target_area = None
        self.whitespace_area = None
        self.mov_node_size_real = None
        self.original_filler_area_total = None
        self.original_pin_rel_cpos = None
        self.original_target_density = None
        self.original_total_mov_area_without_filler = None
        self.original_num_fillers = None
        self.classfied_cell_indices = None
        self.original_mov_node_size = None
        self.original_mov_node_size_real = None
        self.original_init_density_map = None
        self.inflate_ratio_prev = None
        self.delta_inflate_ratio_prev = None
        self.inflate_ratio_from_cg_map_recorder = []
        
route_cache = RouteCache()


def get_route_input_mat():
    return route_cache.route_input_mat


def draw_cg_fig(args, t: torch.Tensor, info, title):
    design_name, iteration, pic_type = info
    filename = "%s_iter%d_%s.png" % (design_name, iteration, pic_type)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path: str = os.path.join(res_root, args.route_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
    
    t_clamp = t.clamp(max=2)
    ratio = t_clamp.shape[1] / t_clamp.shape[0]
    plt.figure(figsize=(6, 5 * ratio))
    colors = [(0.0, "blue"), (0.4, "green"), (0.45, "yellow"), (0.55, "red"), (1.0, "black")]
    cmap = LinearSegmentedColormap.from_list("custom_jet", colors)
    ax = sns.heatmap(t_clamp.t().flip(0).cpu().numpy(), cmap=cmap, vmin=0.0, vmax=2.0)
    
    plt.title(title)
    plt.savefig(png_path)
    plt.close()

def draw_route_gradmat_fig(args, t: torch.Tensor, info, title):
    design_name, iteration, pic_type = info
    filename = "%s_iter%d_%s.png" % (design_name, iteration, pic_type)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path: str = os.path.join(res_root, args.route_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
    
    t_clamp = t
    ratio = t_clamp.shape[1] / t_clamp.shape[0]
    plt.figure(figsize=(6, 5 * ratio))
    if pic_type=="route_gradmatX" or pic_type=="route_gradmatY":
        ax = sns.heatmap(t_clamp.t().flip(0).cpu().numpy(), cmap="bwr")
    else:
        ax = sns.heatmap(t_clamp.t().flip(0).cpu().numpy(), cmap="jet")
    plt.title(title)
    plt.savefig(png_path)
    plt.close()

def evaluate_routability(args, logger, cg_mapHV: torch.Tensor):
    # compute RC
    ace_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ace_tsr = torch.tensor(ace_list, device=cg_mapHV.device, dtype=cg_mapHV.dtype)
    tmp: torch.Tensor = torch.sort((cg_mapHV + 1).reshape(2, -1), descending=True)[0]
    rc = torch.cumsum(tmp, 1) / torch.arange(1, tmp.shape[1] + 1, device=tmp.device, dtype=tmp.dtype)
    indices = (tmp.shape[1] * ace_tsr).long()
    selected_rc = rc[:, indices].cpu()
    log_str = "\n           "
    log_str += "\t".join(["ACE"] + ["%.2f%%" % (i * 100) for i in ace_list]) + "\n           "
    log_str += "\t".join(["HOR"] + ["%.4f" % i for i in selected_rc[0]]) + "\n           "
    log_str += "\t".join(["VER"] + ["%.4f" % i for i in selected_rc[1]])
    logger.info('RC Value:%s' % log_str)

    return ace_list, selected_rc

def calc_gr_wl_via(grdb, routeforce):
    step_x, step_y = routeforce.gcell_steps()
    layer_pitch = routeforce.layer_pitch()
    layer_m2_pitch = layer_pitch[1] if len(layer_pitch) > 1 else layer_pitch[0]

    gr_wirelength, gr_numVias = grdb.report_gr_stat()
    gr_wirelength = gr_wirelength * max(step_x, step_y) / layer_m2_pitch
    
    return gr_wirelength, gr_numVias

def estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map):
    step_x, step_y = routeforce.gcell_steps()
    layer_width = routeforce.layer_width()
    layer_pitch = routeforce.layer_pitch()
    microns = float(routeforce.microns())
    layer_m2_pitch = layer_pitch[1] if len(layer_pitch) > 1 else layer_pitch[0]

    m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
    hId = 1 if m1direction else 0
    vId = 0 if m1direction else 1

    layer_area = torch.tensor(layer_width, device=cap_map.device, dtype=cap_map.dtype)
    layer_area[hId::2].mul_(step_x / microns / layer_m2_pitch / layer_m2_pitch)
    layer_area[vId::2].mul_(step_y / microns / layer_m2_pitch / layer_m2_pitch)
    
    wire_ovfl_map = (wire_dmd_map - cap_map).clamp_(min=0.0)
    routedShortArea = (wire_ovfl_map.sum(dim=(1, 2)) * layer_area).sum()

    via_ovfl_mask = (wire_dmd_map > cap_map).float()
    routedShortViaNum = (via_ovfl_mask * via_dmd_map).sum()

    return (routedShortArea + routedShortViaNum).item()


def get_fft_scale(num_bin_x, num_bin_y, device, scale_w_k=True):
    w_j = (
        torch.arange(num_bin_x, device=device)
        .float()
        .mul(2 * np.pi / num_bin_x)
        .reshape(num_bin_x, 1)
    )
    w_k = (
        torch.arange(num_bin_y, device=device)
        .float()
        .mul(2 * np.pi / num_bin_y)
        .reshape(1, num_bin_y)
    )
    # scale_w_k because the aspect ratio of a bin may not be 1
    # NOTE: we will not scale down w_k in NN since it may distrub the training
    if scale_w_k:
        w_k.mul_(num_bin_y / num_bin_x)
    wj2_plus_wk2 = w_j.pow(2) + w_k.pow(2)
    wj2_plus_wk2[0, 0] = 1.0

    potential_scale = 1.0 / wj2_plus_wk2
    potential_scale[0, 0] = 0.0

    force_x_scale = w_j * potential_scale * 0.5
    force_y_scale = w_k * potential_scale * 0.5

    force_x_coeff = ((-1.0) ** torch.arange(num_bin_x, device=device)).unsqueeze(1)
    force_y_coeff = ((-1.0) ** torch.arange(num_bin_y, device=device)).unsqueeze(0)

    potential_coeff = 1.0

    return (
        potential_scale,
        potential_coeff,
        force_x_scale,
        force_y_scale,
        force_x_coeff,
        force_y_coeff,
    )


def get_route_force(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio, evaluator_fn,
    constraint_fn=None, skip_m1_route=True, enable_filler_grad=True 
):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    _, filler_lhs = data.movable_connected_index
    filler_rhs = mov_node_pos.shape[0]
    num_fillers = filler_rhs - filler_lhs
    num_conn_nodes = (mov_rhs - mov_lhs) + (fix_rhs - fix_lhs)

    mov_route_grad = torch.zeros_like(mov_node_pos)
    mov_congest_grad = torch.zeros_like(mov_node_pos)
    mov_pseudo_grad = torch.zeros_like(mov_node_pos)
    
    # 1) run global routing and compute gradient mat
    grdb, route_input_mat, routeforce, route_gradmat, route_gradmat_upset = None, None, None, None, None
    if ps.recal_conn_route_force:
        ps.recal_conn_route_force = False
        output = run_gr_and_fft_main(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, 
            constraint_fn=constraint_fn, skip_m1_route=skip_m1_route, run_fft=True, rerun_route=ps.rerun_route
        )
        grdb, routeforce, cg_mapAll, route_input_mat, cg_mapHV, map_raw, map_2d, route_gradmat, route_gradmat_upset, gr_metrics = output
        dmd_map, wire_dmd_map, via_dmd_map, cap_map = map_raw
        dmd_map2d, wire_dmd_map2d, via_dmd_map2d, cap_map2d = map_2d
        
        if evaluator_fn is not None:
            hpwl, overflow = evaluator_fn(mov_node_pos)
            ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
        # ------------------------------------------------------------
        # 2) start force computation
        # 2.1) compute routing wire force
        conn_route_grad = conn_route_force(
            num_conn_nodes, cg_mapAll, wire_dmd_map2d, via_dmd_map2d, cap_map2d,
            route_gradmat, routeforce, args, data
        )
        mov_route_grad[mov_lhs:mov_rhs] = conn_route_grad[mov_lhs:mov_rhs]

        route_cache.grdb = grdb
        route_cache.route_input_mat = route_input_mat
        route_cache.routeforce = routeforce
        route_cache.route_gradmat = route_gradmat
        route_cache.route_gradmat_upset = route_gradmat_upset
        route_cache.cg_mapAll = cg_mapAll
        route_cache.mov_route_grad = mov_route_grad
    else:
        grdb = route_cache.grdb
        route_input_mat = route_cache.route_input_mat
        routeforce = route_cache.routeforce
        route_gradmat = route_cache.route_gradmat
        route_gradmat_upset = route_cache.route_gradmat_upset
        cg_mapAll = route_cache.cg_mapAll
        mov_route_grad = route_cache.mov_route_grad
    
    # 2.2.1) compute congestion region force : push cells away from congested area
    mov_congest_grad[:filler_lhs] += cell_congestion_force(
        args, data, mov_node_pos, mov_node_size, expand_ratio,
        cg_mapAll, route_gradmat, routeforce, 0, filler_lhs, -1.0
    )
    if num_fillers > 1:
        mov_congest_grad[filler_lhs:filler_rhs] += cell_congestion_force(
            args, data, mov_node_pos, mov_node_size, expand_ratio,
            cg_mapAll, route_gradmat, routeforce, filler_lhs, filler_rhs, 1.0
        )    
    ps.route_net_force_iter += 1
    if ps.route_net_force_iter<30:
        mov_congest_grad[:filler_lhs] += net_congestion_force(
                    args, data, mov_node_pos, mov_node_size, expand_ratio,
                    cg_mapAll, route_gradmat, routeforce, 0, filler_lhs, -1.0
            )        

    ps.mov_node_to_num_pseudo_pins = torch.zeros_like(mov_node_pos)
    if num_fillers > 0:
        # 2.3) compute pseudo net force, a force to push fillers to congested area
        num_fillers_selected = int(num_fillers * 0.05)
        mov_pseudo_grad[filler_lhs:filler_lhs + num_fillers_selected] += filler_pseudo_wire_force(
            data, ps, mov_node_pos, mov_node_size, routeforce, route_input_mat, filler_lhs, filler_lhs + num_fillers_selected
        )
        ps.mov_node_to_num_pseudo_pins[filler_lhs:filler_lhs + num_fillers_selected] += 1
    else:
        mov_pseudo_grad = None
    
    '''''
    cg_map_medium = torch.zeros(64,64,dtype=torch.float32)
    for i in range(0,64):
        for j in range(0,64):
            cg_map_medium[i][j] =cg_mapAll[i*8:(i+1)*8, j*8:(j+1)*8].sum() / (8 * 8)
            
    ps.cg_map_iter += 1
    if ps.cg_map_iter % 5 == 0 :
        mov_lhs, mov_rhs = data.movable_index
        fix_node_pos_to_draw = data.node_pos[mov_rhs:, ...].clone()
        fix_node_size_to_draw = data.node_size[mov_rhs:, ...].clone()
    
        draw_cg_fig_with_cairo(
            cg_map=cg_mapAll,
            args=args,
            fix_node_pos=fix_node_pos_to_draw,
            fix_node_size=fix_node_size_to_draw,
            data=data,
            name=data.design_name,
            iter=ps.cg_map_iter,
            route_grad=mov_congest_grad
        )
'''
    return mov_route_grad, mov_congest_grad, mov_pseudo_grad


def run_gr_and_fft_main(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=None,rerun_route=False, **kwargs
):
    if rerun_route:
        ps.rerun_route = False
        logger.info("Update gpdb node pos...")
        mov_lhs, mov_rhs = data.movable_index
        mov_node_pos = constraint_fn(mov_node_pos)
        node_pos = mov_node_pos[mov_lhs:mov_rhs]
        node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
        exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift)
        exact_node_lpos = torch.round(exact_node_pos - torch.round(data.node_size * data.die_scale) / 2).cpu()
        gpdb.apply_node_lpos(exact_node_lpos)

        output = run_gr_and_fft(args, logger, data, rawdb, gpdb, ps, mov_node_pos[mov_lhs:mov_rhs], **kwargs)
        route_cache.gr_and_fft_main_output= output

    else:
        output = route_cache.gr_and_fft_main_output    
        
    return output


def run_gr_and_fft(args, logger, data, rawdb, gpdb, ps, mov_node_pos=None, grdb=None, skip_m1_route=True, run_fft=False, visualize=True, report_gr_metrics_only=False, given_gr_params={}):
    route_size = 512
    iteration = ps.iter - 1  # ps.iter is increased before running GR optimization
    die_ratio = (data.__ori_die_hx__ - data.__ori_die_lx__) / (data.__ori_die_hy__ - data.__ori_die_ly__)
    route_xSize = route_size if die_ratio <= 1 else round(route_size / die_ratio)
    route_ySize = route_size if die_ratio >= 1 else round(route_size / die_ratio)

    # 1.1) init GRDatabase
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    gr_params = {"device_id": args.gpu, "route_xSize": route_xSize, "route_ySize": route_ySize}
    gr_params.update(given_gr_params)
    gpugr.load_gr_params(gr_params)

    # 1.2) do GR
    logger.info("--------- Start GR in Iter: %d ---------" % iteration)
    if grdb is None:
        grdb = gpugr.create_grdatabase(rawdb, gpdb)
    routeforce = gpugr.create_routeforce(grdb)
    routeforce.run_ggr()
    logger.info("--------- End GR ---------")

    m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
    hId = 1 if m1direction else 0
    vId = 0 if m1direction else 1
    aId = 0
    if skip_m1_route:
        aId = 1
        hId = hId + 2 if hId == 0 else hId
        vId = vId + 2 if vId == 0 else vId

    # 1.3) calculate capacity and demand map
    dmd_map, wire_dmd_map, via_dmd_map = routeforce.dmd_map()
    cap_map: torch.Tensor = routeforce.cap_map()

    dmd_map2d: torch.Tensor = dmd_map[aId:].sum(dim=0)
    wire_dmd_map2d: torch.Tensor = wire_dmd_map[aId:].sum(dim=0)
    via_dmd_map2d: torch.Tensor = via_dmd_map[aId:].sum(dim=0)
    cap_map2d: torch.Tensor = cap_map[aId:].sum(dim=0)

    cg_mapAll = dmd_map2d / cap_map2d  # ignore metal0
    min_cg = torch.tensor(0.8, dtype=torch.float).to(cg_mapAll.device)  # set too high may cause force_x(y)_map error i.e. route_force direction error
    route_input_mat = torch.where(cg_mapAll > min_cg, cg_mapAll, min_cg)#TODO modify:route_input_mat
    route_input_mat = torch.where(route_input_mat > 1.0, torch.pow(route_input_mat, 2), route_input_mat)
    route_input_mat_upset = cg_mapAll.clamp_(max=2.0,min=0.0)
    route_input_mat_upset = 2.05 - route_input_mat_upset
    route_input_mat_upset = torch.where(route_input_mat_upset > min_cg, route_input_mat_upset, min_cg)
    route_input_mat_upset = torch.where(route_input_mat_upset > 1.0, torch.pow(route_input_mat_upset, 2), route_input_mat_upset)

    cg_mapH = dmd_map[hId::2].sum(dim=0) / cap_map[hId::2].sum(dim=0)
    cg_mapV = dmd_map[vId::2].sum(dim=0) / cap_map[vId::2].sum(dim=0)
    cg_mapHV = torch.stack((cg_mapH, cg_mapV))
    min_cg_mapHV = torch.tensor(0.0, dtype=torch.float).to(cg_mapHV.device)
    cg_mapHV = torch.where(cg_mapHV > 1, cg_mapHV - 1, min_cg_mapHV)

    cgOvfl = (dmd_map[aId:] / cap_map[aId:]).max(dim=0)[0].clamp(min=1.0) - 1
    map_raw = (dmd_map, wire_dmd_map, via_dmd_map, cap_map)
    map_2d = (dmd_map2d, wire_dmd_map2d, via_dmd_map2d, cap_map2d)
    
    #set_macro_region_congestion_one(cg_mapAll, data, routeforce)
    #handle_macro_margin_init_density(cg_mapAll, data, args, routeforce)

    # 1.4) compute congestion map's gradient
    route_gradmat = None
    if run_fft:
        fft_scale = get_fft_scale(route_input_mat.shape[0], route_input_mat.shape[1], device)
        potential_scale, _, force_x_scale, force_y_scale, _, _ = fft_scale
        fft_coeff = dct2(route_input_mat)
        force_x_map = idxst_idct(fft_coeff * force_x_scale)
        force_y_map = idct_idxst(fft_coeff * force_y_scale)
        potential_map = idct2(fft_coeff * potential_scale)
        route_gradmat = torch.vstack(
            (force_x_map.unsqueeze(0), force_y_map.unsqueeze(0))
        ).contiguous()  # 2 x M x N
    if run_fft:
        fft_scale = get_fft_scale(route_input_mat_upset.shape[0], route_input_mat_upset.shape[1], device)
        potential_scale, _, force_x_scale, force_y_scale, _, _ = fft_scale
        fft_coeff = dct2(route_input_mat_upset)
        force_x_map_upset = idxst_idct(fft_coeff * force_x_scale)
        force_y_map_upset = idct_idxst(fft_coeff * force_y_scale)
        potential_map = idct2(fft_coeff * potential_scale)
        route_gradmat_upset = torch.vstack(
            (force_x_map_upset.unsqueeze(0), force_y_map_upset.unsqueeze(0))
        ).contiguous()  # 2 x M x N
        

    # 1.5) print stat
    logger.info(
        "cgMap max: %.4f mean: %.4f std: %.4f | cgOvfl max: %.4f mean: %.4f std: %.4f"
        % (
            cg_mapAll.max().item(),
            cg_mapAll.mean().item(),
            cg_mapAll.std().item(),
            cgOvfl.max().item(),
            cgOvfl.mean().item(),
            cgOvfl.std().item(),
        )
    )
    logger.info(
        "cgMapH max: %.4f mean: %.4f std: %.4f | cgMapV max: %.4f mean: %.4f std: %.4f"
        % (
            cg_mapHV[0].max().item(),
            cg_mapHV[0].mean().item(),
            cg_mapHV[0].std().item(),
            cg_mapHV[1].max().item(),
            cg_mapHV[1].mean().item(),
            cg_mapHV[1].std().item(),
        )
    )

    numOvflNets = routeforce.num_ovfl_nets()
    gr_wirelength, gr_numVias = calc_gr_wl_via(grdb, routeforce)
    gr_numShorts = estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map)

    ace_list, selected_rc = evaluate_routability(args, logger, cg_mapHV)
    rc_hor_mean, rc_ver_mean = selected_rc.mean(dim=1).tolist()

    logger.info(
        "#OvflNets: %d (%.2f%%), GR WL: %d, GR #Vias: %d, #EstShorts: %d, RC Hor: %.3f, RC Ver: %.3f, overflow: %.3f" % (
            numOvflNets,
            numOvflNets / data.num_nets * 100,
            gr_wirelength,
            gr_numVias,
            gr_numShorts,
            rc_hor_mean,
            rc_ver_mean,
            ps.recorder.overflow[-1]
        )
    )

    gr_metrics = (numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean)

    if args.draw_congestion_map:
        title = "#OvflNets: %.2e, WL: %.2e, #Vias: %.2e\n#Shorts: %.2e, RC Hor: %.3f, RC Ver: %.3f " % (
            numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean
        )
        draw_cg_fig(args, cg_mapAll, (data.design_name, iteration, "cg_mapAll"), title)
        
    if report_gr_metrics_only:
        return gr_metrics

    return grdb, routeforce, cg_mapAll, route_input_mat, cg_mapHV, map_raw, map_2d, route_gradmat, route_gradmat_upset, gr_metrics


def conn_route_force(
    num_conn_nodes, cg_mapAll, wire_dmd_map2d, via_dmd_map2d, cap_map2d,
    route_gradmat, routeforce, args, data
):
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    max_n_grid = max(cg_mapAll.shape[0], cg_mapAll.shape[1])
    mask_map = (cg_mapAll > 1.0).float()
    # mask_map = torch.logical_or(cg_mapH > 1.5, cg_mapV > 1.5).float()
    dist_weights = torch.ones(max_n_grid + 2, device=device)
    wirelength_weights = torch.ones(max_n_grid + 2, device=device)
    wirelength_weights[20:] = 0#TODO:modify:long path be ignored
    unit_wire_cost = 1.0
    unit_via_cost = 1.0
    grad_weight = -1.0

    conn_route_grad: torch.Tensor = routeforce.route_grad(
        mask_map,
        wire_dmd_map2d,
        via_dmd_map2d,
        cap_map2d,
        dist_weights,
        wirelength_weights,
        route_gradmat,
        data.node2pin_list,
        data.node2pin_list_end,
        grad_weight,
        unit_wire_cost,
        unit_via_cost,
        num_conn_nodes,
    )
        
    return conn_route_grad

def net_congestion_force(
    args, data, mov_node_pos, mov_node_size, expand_ratio, cg_mapAll, route_gradmat, routeforce, lhs, rhs, grad_weight=1.0
):
    
    # NOTE: grad_weight == 1.0, push cell to congested area
    num_bin_x, num_bin_y = route_gradmat.shape[1], route_gradmat.shape[2]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width
    selected_nets = (data.net_to_num_pins < 5).float() 
    
    net_center_pos = torch.zeros((data.net_to_num_pins.shape[0], 2), device=mov_node_pos.device)

    net_center_pos = routeforce.calc_net_center_pos(
        mov_node_pos[lhs:rhs],
        data.pin_id2node_id,
        data.pin_rel_cpos,
        data.hyperedge_list,
        data.hyperedge_list_end,
        selected_nets,
        data.net_mask
    )
    net_to_num_pins_expanded = data.net_to_num_pins.unsqueeze(1).expand(-1, 2)
    net_center_size = torch.ones_like(net_center_pos) * net_to_num_pins_expanded
    net_center_size = torch.clamp(net_center_size, max=5)
    max_num_pin =torch.clamp(data.net_to_num_pins, max=5)
    net_expend_ratio = torch.ones_like(selected_nets)/max_num_pin 
    net_center_grad = routeforce.filler_route_grad(
        net_center_pos,
        net_center_size,
        selected_nets,
        net_expend_ratio,
        route_gradmat,
        grad_weight,
        unit_len_x,
        unit_len_y,
        num_bin_x,
        num_bin_y,
        data.num_nets
    )        
    net2node_grad = torch.zeros_like(mov_node_pos[lhs:rhs])
    net2node_grad = routeforce.net_to_node_force(
        net_center_grad,
        data.hyperedge_list,
        data.hyperedge_list_end,
        data.node2pin_list,
        data.node2pin_list_end,
        data.pin_id2node_id,
        selected_nets,
        rhs
    )
    
    return net2node_grad

def cell_congestion_force(
    args, data, mov_node_pos, mov_node_size, expand_ratio, cg_mapAll, route_gradmat, routeforce, lhs, rhs, grad_weight=1.0
):
    # NOTE: grad_weight == 1.0, push cell to congested area
    num_bin_x, num_bin_y = route_gradmat.shape[1], route_gradmat.shape[2]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width
    
    # value > 1.0 means congested
    node_cg_value = calc_node_congestion_value(cg_mapAll, mov_node_pos, num_bin_x, num_bin_y, unit_len_x, unit_len_y)
    if grad_weight > 0: # for filler
        min_cg_one = torch.tensor(0.0, dtype=torch.float).to(cg_mapAll.device)
        filler_weight = torch.where((node_cg_value > 0.5) & (node_cg_value < 1.0), node_cg_value*1.5, min_cg_one)
    else: # for movable cells
        high_indices, medium_indices, low_indices = classify_cell_according_pins(data.mov_node_to_num_pins, lhs, rhs)
        min_cg_one = torch.tensor(0.7, dtype=torch.float).to(cg_mapAll.device)        
        node_cg_value = node_cg_value.float()
        filler_weight = torch.where(node_cg_value > min_cg_one, node_cg_value, torch.tensor(0.0, dtype=torch.float).to(cg_mapAll.device))
        filler_weight[medium_indices] = 0.8
        filler_weight[low_indices] = 0.0
        filler_weight.clamp_(max=2.0,min=0)
    
    filler_route_grad = routeforce.filler_route_grad(
        mov_node_pos[lhs:rhs],
        mov_node_size[lhs:rhs],
        filler_weight,
        expand_ratio[lhs:rhs],
        route_gradmat,
        grad_weight,
        unit_len_x,
        unit_len_y,
        num_bin_x,
        num_bin_y,
        rhs - lhs
    )
    
    filler_route_grad =torch.where(filler_route_grad.abs() > 1e-3, filler_route_grad, torch.tensor(0.0, dtype=torch.float).to(filler_route_grad.device))
    
    return filler_route_grad
    
def filler_pseudo_wire_force(data, ps, mov_node_pos, mov_node_size, routeforce, cg_mapAll, lhs, rhs):
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=2)
    cg_mapAll_blurred: torch.Tensor = blurrer(cg_mapAll.unsqueeze(0)).squeeze(0)
    meanKrnl = 11
    cg_mapMean = torch.nn.functional.avg_pool2d(cg_mapAll.unsqueeze(0), meanKrnl, 1, padding=meanKrnl // 2).squeeze(0)
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    pseudo_pin_pos = (cg_mapMean == torch.max(cg_mapMean)).nonzero()[0].float() + 0.5
    scale = torch.tensor([unit_len_x, unit_len_y], device=mov_node_pos.device)
    pseudo_pin_pos.mul_(scale)
    pseudo_pin_pos = pseudo_pin_pos.repeat(rhs - lhs, 1)
    pseudo_pin_pos.add_(torch.randn_like(pseudo_pin_pos) * scale * 5)

    pseudo_grad = routeforce.pseudo_grad(mov_node_pos[lhs:rhs], pseudo_pin_pos, ps.wa_coeff)

    return pseudo_grad

def route_inflation_Xplace_method(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
    constraint_fn=None, skip_m1_route=True, use_weighted_inflation=True, hv_same_ratio=True,
    dynamic_target_density=True, rerun_route=False, **kwargs
):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    _, filler_lhs = data.movable_connected_index
    filler_rhs = mov_node_pos.shape[0]
    num_fillers = filler_rhs - filler_lhs
    decrease_target_density = False

    # FIXME: How can we dynamically set args.min_area_inc according to the congestion map?
    #        I believe there should be elegant ways to do that like computing local statistics...
    if args.design_name == "ispd18_test10":
        # This design is extremely congested in its center.
        # Temporarily hard coded. This value may not be the best...
        args.min_area_inc = 0.001

    if route_cache.first_run:
        # TODO: check args.target_density should be changed or not?
        route_cache.original_mov_node_size = mov_node_size.clone()
        # NOTE: size_real denotes the node size before node expand
        # these size_real are internally maintained by route_cache 
        route_cache.mov_node_size_real = data.mov_node_size_real.clone()
        route_cache.original_mov_node_size_real = data.mov_node_size_real.clone()
        mov_node_size_real = route_cache.mov_node_size_real

        mov_conn_size = mov_node_size_real[mov_lhs:filler_lhs, ...]
        filler_size = mov_node_size_real[filler_lhs:filler_rhs, ...]

        route_cache.first_run = False
        tmp_area = torch.sum(torch.prod(mov_node_size_real, 1)).item()
        route_cache.target_area = tmp_area
        route_cache.placeable_area = tmp_area / args.target_density
        route_cache.whitespace_area = route_cache.placeable_area - torch.sum(torch.prod(mov_conn_size, 1)).item()
        route_cache.original_num_fillers = filler_rhs - filler_lhs
        route_cache.original_filler_area_total = torch.sum(torch.prod(filler_size, 1)).item()
        route_cache.original_pin_rel_cpos = data.pin_rel_cpos.clone()
        route_cache.original_target_density = copy.deepcopy(args.target_density)
        route_cache.original_total_mov_area_without_filler = copy.deepcopy(data.__total_mov_area_without_filler__)
        route_cache.original_init_density_map = data.init_density_map.clone()

    ori_mov_node_size = route_cache.original_mov_node_size
    ori_mov_node_size_real = route_cache.original_mov_node_size_real
    mov_node_size_real = route_cache.mov_node_size_real

    # 1) check remain space
    last_mov_area = torch.prod(mov_node_size_real[mov_lhs:filler_lhs], 1)
    last_mov_area_total = last_mov_area.sum().item()
    last_filler_area_total = torch.sum(torch.prod(mov_node_size_real[filler_lhs:filler_rhs], 1)).item()
    max_inc_area_total = min(0.1 * route_cache.whitespace_area, route_cache.placeable_area - last_mov_area_total) # TODO: tune
    if max_inc_area_total <= 0:
        logger.warning("No space to inflate. Terminate inflation.")
        ps.use_cell_inflate = False  # not inflation anymore
        return None
    
    # 2) run GR to get congestion map
    output = run_gr_and_fft_main(
        args, logger, data, rawdb, gpdb, ps, mov_node_pos, 
        constraint_fn=constraint_fn, skip_m1_route=skip_m1_route, rerun_route=rerun_route,
        run_fft=ps.open_route_force_opt, **kwargs
    )
    grdb, routeforce, cg_mapAll, _, cg_mapHV, _, _, route_gradmat, gr_metrics = output
    numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean = gr_metrics
    cg_map_inflation = torch.where(cg_mapAll > 1, cg_mapAll - 1, 0)
    num_bin_x, num_bin_y = cg_map_inflation.shape[0], cg_map_inflation.shape[1]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    # 3) get next step inflation ratio   
    if hv_same_ratio:
        inflate_mat: torch.Tensor = torch.stack((cg_map_inflation + 1, cg_map_inflation + 1)).contiguous().pow_(2)
    else:
        inflate_mat: torch.Tensor = (cg_mapHV + 1).permute(0, 2, 1).contiguous()
    if dynamic_target_density and last_filler_area_total / last_mov_area_total > 1.1 and numOvflNets / data.num_nets > 0.04: # TODO: the if condition can't be satisfied in all cases(ispd2015)
        # filler area too large => low utilization, can adjust size more aggressively to reduce GR overflow
        max_inc_area_total = route_cache.placeable_area - last_mov_area_total
        logger.info("Low utilization detect, globally inflate...")
        global_ratio = (rc_hor_mean + rc_ver_mean) / 2
        inflate_mat *= global_ratio
        decrease_target_density = True
    inflate_mat.clamp_(min=1.0, max=2.0)

    # NOTE: 1) If use_weighted_inflation == False, use max congestion as inflation ratio.
    #       2) We use mov_node_size instead of mov_node_size_real to cover more GR Grids.
    mov_node_weight = torch.ones(mov_node_size.shape[0], device=mov_node_size.device)
    this_mov_conn_inflate_ratio = routeforce.inflate_ratio(
        mov_node_pos[mov_lhs:filler_lhs],
        mov_node_size[mov_lhs:filler_lhs],
        mov_node_weight[mov_lhs:filler_lhs],
        torch.ones_like(mov_node_weight[mov_lhs:filler_lhs]),  # use expand ratio == 1
        inflate_mat,
        1.0, unit_len_x, unit_len_y, num_bin_x, num_bin_y, use_weighted_inflation
    )
    if hv_same_ratio:
        this_mov_conn_inflate_ratio.sqrt_()
    else:
        this_mov_conn_inflate_ratio.sqrt_()

    # 4.1) if the remaining space is not enough, scale down the movable cell inflation ratio
    expect_new_mov_area = torch.prod(this_mov_conn_inflate_ratio * mov_node_size_real[mov_lhs:filler_lhs], 1)
    inc_mov_area = expect_new_mov_area - last_mov_area
    inc_mov_area_total = inc_mov_area.sum().item()
    inc_area_scale = max_inc_area_total / inc_mov_area_total
    if inc_mov_area_total <= 0:
        logger.warning("Negative area increment %.4f. Early terminate cell inflation." % (inc_mov_area_total))
        ps.use_cell_inflate = False  # not inflation anymore
        return gr_metrics, None, None
    if inc_area_scale < 1:
        logger.warning("Not enough space to inflate. Scale down.")
        inc_mov_area *= inc_area_scale
        inc_mov_area_total *= inc_area_scale
        new_mov_area = inc_mov_area + last_mov_area
        size_scale = (new_mov_area / expect_new_mov_area).sqrt_().unsqueeze_(1)
        this_mov_conn_inflate_ratio *= size_scale
    else:
        new_mov_area = expect_new_mov_area
    # 4.2) update total mov inflation ratio
    new_mov_node_size_real: torch.Tensor = mov_node_size_real.clone()
    new_mov_node_size_real[mov_lhs:filler_lhs].mul_(this_mov_conn_inflate_ratio)

    if inc_mov_area_total / last_mov_area_total < args.min_area_inc:
        logger.warning(
            "Too small relative area increment (%.4f < %.4f). Early terminate cell inflation." % (
                inc_mov_area_total / last_mov_area_total, args.min_area_inc
        ))
        ps.use_cell_inflate = False  # not inflation anymore
        return gr_metrics, None, None

    # 5) update total filler inflation ratio
    new_mov_area_total = last_mov_area_total + inc_mov_area_total
    new_filler_area_total = 0.0
    filler_scale = 0.0
    if new_mov_area_total + last_filler_area_total > route_cache.target_area:
        new_filler_area_total = max(route_cache.target_area - new_mov_area_total, 0)
        if decrease_target_density:
            logger.info("Removing some pre-inserted fillers / FloatMov nodes to decrease the target density...")
            # standard cell density: 1 / (1 + 1.1) = 0.4762
            new_target_density = max(0.4762, 0.85 * args.target_density)
            new_target_area = new_target_density * route_cache.placeable_area
            new_filler_area_total = max(new_target_area - new_mov_area_total, 0)
            filler_scale = new_filler_area_total / route_cache.original_filler_area_total
            original_num_fillers = route_cache.original_num_fillers
            num_remain_cells = filler_lhs + math.ceil(filler_scale * original_num_fillers)

            route_cache.target_area = new_target_area
            # set filler size as 0 to remove
            new_mov_node_size_real[num_remain_cells:] = 0
            mov_node_size_real[num_remain_cells:] = 0
        elif route_cache.original_filler_area_total > 0:
            filler_scale = math.sqrt(new_filler_area_total / route_cache.original_filler_area_total)
            new_mov_node_size_real[filler_lhs:filler_rhs] = filler_scale * ori_mov_node_size_real[filler_lhs:filler_rhs]
    else:
        new_filler_area_total = last_filler_area_total

    # pin rel cpos should be scaled by real inflate_ratio
    inflate_ratio = new_mov_node_size_real / mov_node_size_real
    new_pin_rel_cpos = routeforce.inflate_pin_rel_cpos(
        inflate_ratio,
        route_cache.original_pin_rel_cpos,
        data.pin_id2node_id,
        mov_rhs - filler_lhs
    )

    mov_node_size_real.copy_(new_mov_node_size_real)
    data.pin_rel_cpos.copy_(new_pin_rel_cpos)
    old_target_density = copy.deepcopy(args.target_density)
    args.target_density = (new_mov_area_total + new_filler_area_total) / route_cache.placeable_area
    logger.info("Update target density from %.4f to %.4f" % (old_target_density, args.target_density))

    if decrease_target_density:
        # since we decrease target density, update the init density map correspondingly
        logger.warning("Remove nodes to reduce target density from %.4f to %.4f. This step may remove some FloatMov. #Nodes from %d to %d" % 
            (old_target_density, args.target_density, filler_rhs, num_remain_cells))
        data.init_density_map.clamp_(min=0.0, max=1.0).div_(old_target_density).mul_(args.target_density)

    logger.info("Absolute area (last) | mov: %.4E, filler: %.4E, all_cells: %.4E" % (
        last_mov_area_total, last_filler_area_total, last_mov_area_total + last_filler_area_total
    ))
    logger.info("Absolute area (this) | mov: %.4E, filler: %.4E, all_cells: %.4E" % (
        new_mov_area_total, new_filler_area_total, new_mov_area_total + new_filler_area_total
    ))
    logger.info("Relative area change | increment: %.4f, mov: %.4f, filler: %.4f, all_cells: %.4f" % (
        inc_mov_area_total / last_mov_area_total,
        new_mov_area_total / last_mov_area_total,
        new_filler_area_total / last_filler_area_total if last_filler_area_total > 1e-5 else 0,
        (new_mov_area_total + new_filler_area_total) / (last_mov_area_total + last_filler_area_total)
    ))
    logger.info("Inflation Rate | movable: avgX/maxX %.4f/%.4f avgY/maxY %.4f/%.4f, filler: %.4f" % (
        inflate_ratio[mov_lhs:filler_lhs, 0].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 0].max().item(),
        inflate_ratio[mov_lhs:filler_lhs, 1].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 1].max().item(),
        filler_scale
    ))

    if abs(args.target_density * route_cache.placeable_area - torch.sum(torch.prod(mov_node_size_real, 1)).item()) > 1.0:
        logger.warning("Please check inflation...")
        logger.info("new_total_mov_area: %.1f ori_total_mov_area: %.1f placeable_area: %.1f target_density: %.2f placeable_area * target_density: %.1f" % (
            torch.sum(torch.prod(mov_node_size_real, 1)).item(),
            torch.sum(torch.prod(ori_mov_node_size, 1)).item(),
            route_cache.placeable_area,
            args.target_density,
            args.target_density * route_cache.placeable_area
        ))
    
    route_cache.mov_node_size_real = mov_node_size_real
    data.mov_node_area = torch.prod(mov_node_size_real, 1).unsqueeze_(1)

    mov_node_size = mov_node_size_real
    expand_ratio = mov_node_size.new_ones((mov_node_size.shape[0]))
    if args.clamp_node:
        mov_node_area = torch.prod(mov_node_size, 1)
        clamp_mov_node_size = mov_node_size.clamp(min=data.unit_len * math.sqrt(2))
        clamp_mov_node_area = torch.prod(clamp_mov_node_size, 1)
        # update
        expand_ratio = mov_node_area / clamp_mov_node_area
        mov_node_size = clamp_mov_node_size

    if decrease_target_density:
        # FIXME: should we update total_mov_area_without_filler when decrease_target_density == False?
        mov_cell_area = torch.prod(new_mov_node_size_real[mov_lhs:mov_rhs, ...], 1)
        data.__total_mov_area_without_filler__ = torch.sum(mov_cell_area).item()

    ps.use_cell_inflate = True
    return gr_metrics, mov_node_size, expand_ratio

def route_inflation(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
    constraint_fn=None, skip_m1_route=True, use_weighted_inflation=True, hv_same_ratio=True,
    dynamic_target_density=True, rerun_route=False, momentum=True, **kwargs
):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    _, filler_lhs = data.movable_connected_index
    filler_rhs = mov_node_pos.shape[0]
    num_fillers = filler_rhs - filler_lhs
    decrease_target_density = False

    # FIXME: How can we dynamically set args.min_area_inc according to the congestion map?
    #        I believe there should be elegant ways to do that like computing local statistics...
    if args.design_name == "ispd18_test10":
        # This design is extremely congested in its center.
        # Temporarily hard coded. This value may not be the best...
        args.min_area_inc = 0.001

    if route_cache.first_run:
        initialize_route_cache(route_cache, data, args, mov_node_size, mov_lhs, filler_lhs, filler_rhs)
        route_cache.first_run = False

    ori_mov_node_size_real = route_cache.original_mov_node_size_real
    ori_mov_node_area = torch.prod(ori_mov_node_size_real, 1)
    mov_node_size_real = route_cache.mov_node_size_real
    # mov_node_size_real = route_cache.original_mov_node_size_real * route_cache.inflate_ratio_prev

    # 1) check remain space
    last_mov_area = torch.prod(mov_node_size_real[mov_lhs:filler_lhs], 1)
    last_mov_area_total = last_mov_area.sum().item()
    last_filler_area_total = torch.sum(torch.prod(mov_node_size_real[filler_lhs:filler_rhs], 1)).item()
    max_inc_area_total = min(0.1 * route_cache.whitespace_area, route_cache.placeable_area - last_mov_area_total) # TODO: tune
    if max_inc_area_total <= 0:
        logger.warning("No space to inflate. Terminate inflation.")
        ps.use_cell_inflate = False  # not inflation anymore
        return None
    
    # 2) run GR to get congestion map
    output = run_gr_and_fft_main(
        args, logger, data, rawdb, gpdb, ps, mov_node_pos, 
        constraint_fn=constraint_fn, skip_m1_route=skip_m1_route, rerun_route=rerun_route,
        run_fft=ps.open_route_force_opt, **kwargs
    )
    grdb, routeforce, cg_mapAll, _, cg_mapHV, _, _, route_gradmat,_, gr_metrics = output
    numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean = gr_metrics
    min_cg_map_inflation = torch.tensor(0.0, dtype=torch.float).to(cg_mapAll.device)
    cg_map_inflation = torch.where(cg_mapAll > 1, cg_mapAll - 1, min_cg_map_inflation)
    num_bin_x, num_bin_y = cg_map_inflation.shape[0], cg_map_inflation.shape[1]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    # 3.1) get inflation ratio according to congestion map
    if hv_same_ratio:
        inflate_mat: torch.Tensor = torch.stack((cg_map_inflation + 1, cg_map_inflation + 1)).contiguous().pow_(2.5)
    else:
        inflate_mat: torch.Tensor = (cg_mapHV + 1).permute(0, 2, 1).contiguous()
    if dynamic_target_density and last_filler_area_total / last_mov_area_total > 1.1 and numOvflNets / data.num_nets > 0.04: # TODO: the if condition can't be satisfied in all cases(ispd2015)
        # filler area too large => low utilization, can adjust size more aggressively to reduce GR overflow
        max_inc_area_total = route_cache.placeable_area - last_mov_area_total
        logger.info("Low utilization detect, globally inflate...")
        global_ratio = (rc_hor_mean + rc_ver_mean) / 2
        inflate_mat *= global_ratio
        decrease_target_density = True
    inflate_mat.clamp_(min=1.0, max=2.5)

    # NOTE: 1) If use_weighted_inflation == False, use max congestion as inflation ratio.
    #       2) We use mov_node_size instead of mov_node_size_real to cover more GR Grids.
    mov_node_weight = torch.ones(mov_node_size.shape[0], device=mov_node_size.device)
    inflate_ratio_from_cg_map = routeforce.inflate_ratio(
        mov_node_pos[mov_lhs:filler_lhs],
        mov_node_size[mov_lhs:filler_lhs],
        mov_node_weight[mov_lhs:filler_lhs],
        torch.ones_like(mov_node_weight[mov_lhs:filler_lhs]),  # use expand ratio == 1
        inflate_mat,
        1.0, unit_len_x, unit_len_y, num_bin_x, num_bin_y, use_weighted_inflation
    )
    inflate_ratio_from_cg_map.sqrt_()
    outlier_count = (inflate_ratio_from_cg_map < 0.99).sum().item()
    if outlier_count > 0:
        logger.warning("Inflation ratio < 1.0 detected: %d" % outlier_count)
    inflate_ratio_from_cg_map.clamp_(min=1.0, max=2.5)
    route_cache.inflate_ratio_from_cg_map_recorder.append(inflate_ratio_from_cg_map)
    
    # 3.2) get next step inflation ratio according to momentum cell inflation
    if momentum:
        # 3.2.1) calculate cell criticality and momentum term
        if route_cache.first_cell_inflate_run: 
            # initialize
            cell_criticality = cal_cell_inflation_criticality(inflate_ratio_from_cg_map, initialize=True)
            route_cache.delta_inflate_ratio_prev = cell_criticality
            route_cache.first_cell_inflate_run = False
        else:
            cell_criticality = cal_cell_inflation_criticality(inflate_ratio_from_cg_map)
            history_weight = 0.6
            route_cache.delta_inflate_ratio_prev = history_weight * route_cache.delta_inflate_ratio_prev + (1 - history_weight) * cell_criticality

        # 3.2.2) update inflation ratio
        inflate_ratio_curr = route_cache.inflate_ratio_prev + route_cache.delta_inflate_ratio_prev
        outlier_count = (inflate_ratio_curr < 0.99).sum().item()
        if outlier_count > 0:
            logger.warning("inflate_ratio_curr_before_scale down < 1.0 detected: %d" % outlier_count)
        outlier_count = (inflate_ratio_curr > 2.5).sum().item()
        if outlier_count > 0:
            logger.warning("inflate_ratio_curr_before_scale down > 2.5 detected: %d" % outlier_count)
        inflate_ratio_curr.clamp_(min=1.0, max=2.5)
        this_mov_conn_inflate_ratio = inflate_ratio_curr / route_cache.inflate_ratio_prev

    else:
        this_mov_conn_inflate_ratio = inflate_ratio_from_cg_map.clone()
        inflate_ratio_curr = inflate_ratio_from_cg_map * mov_node_size_real[mov_lhs:filler_lhs] / ori_mov_node_size_real[mov_lhs:filler_lhs]
    
    # 4.1) if the remaining space is not enough, scale down the movable cell inflation ratio
    expect_new_mov_area = torch.prod(inflate_ratio_curr * ori_mov_node_size_real[mov_lhs:filler_lhs], 1)
    expect_new_mov_area_total = expect_new_mov_area.sum().item()
    inc_mov_area_total = expect_new_mov_area_total - last_mov_area_total
    if inc_mov_area_total <= 0:
        logger.warning("Negative area increment %.4f. Early terminate cell inflation." % (inc_mov_area_total))
           
    mask = inflate_ratio_curr > 1.0
    row_mask = mask.all(dim=1)
    inflate_indices = torch.nonzero(row_mask)    
    # NOTE: all means the all iteration, not the current iteration
    all_max_inc_mov_area_total = last_mov_area_total + max_inc_area_total - route_cache.original_total_mov_area_without_filler
    all_inc_mov_area = (expect_new_mov_area - ori_mov_node_area[mov_lhs:filler_lhs])[inflate_indices]
    all_inc_area_total = all_inc_mov_area.sum().item()
    inc_area_scale = all_max_inc_mov_area_total / all_inc_area_total
    if inc_area_scale < 1:
        logger.warning("Not enough space to inflate. Scale down.")
        all_inc_mov_area *= inc_area_scale
        all_inc_area_total *= inc_area_scale
        new_mov_area = all_inc_mov_area + ori_mov_node_area[mov_lhs:filler_lhs][inflate_indices]
        size_scale = (new_mov_area / expect_new_mov_area[inflate_indices]).sqrt_().unsqueeze_(1)
        inflate_ratio_curr[inflate_indices] *= size_scale
    
    # update inflation ratio for next iteration
    route_cache.inflate_ratio_prev = inflate_ratio_curr
        
    # 4.2) update total mov inflation ratio
    new_mov_node_size_real: torch.Tensor = ori_mov_node_size_real.clone()
    new_mov_node_size_real[mov_lhs:filler_lhs].mul_(inflate_ratio_curr)
    new_mov_area = torch.prod(new_mov_node_size_real[mov_lhs:filler_lhs], 1)
    new_mov_area_total = new_mov_area.sum().item()
    inc_mov_area_total = new_mov_area_total - last_mov_area_total

    if inc_mov_area_total / last_mov_area_total < args.min_area_inc:
        logger.warning(
            "Too small relative area increment (%.4f < %.4f). Early terminate cell inflation." % (
                inc_mov_area_total / last_mov_area_total, args.min_area_inc
        ))
        ps.use_cell_inflate = False  # not inflation anymore
        return gr_metrics, None, None

    # 5) update total filler inflation ratio
    new_filler_area_total = 0.0
    filler_scale = 0.0
    if new_mov_area_total + last_filler_area_total > route_cache.target_area:
        new_filler_area_total = max(route_cache.target_area - new_mov_area_total, 0)
        if decrease_target_density:
            logger.info("Removing some pre-inserted fillers / FloatMov nodes to decrease the target density...")
            # standard cell density: 1 / (1 + 1.1) = 0.4762
            new_target_density = max(0.4762, 0.85 * args.target_density)
            new_target_area = new_target_density * route_cache.placeable_area
            new_filler_area_total = max(new_target_area - new_mov_area_total, 0)
            filler_scale = new_filler_area_total / route_cache.original_filler_area_total
            original_num_fillers = route_cache.original_num_fillers
            num_remain_cells = filler_lhs + math.ceil(filler_scale * original_num_fillers)

            route_cache.target_area = new_target_area
            # set filler size as 0 to remove
            new_mov_node_size_real[num_remain_cells:] = 0
            mov_node_size_real[num_remain_cells:] = 0
        elif route_cache.original_filler_area_total > 0:
            filler_scale = math.sqrt(new_filler_area_total / route_cache.original_filler_area_total)
            new_mov_node_size_real[filler_lhs:filler_rhs] = filler_scale * ori_mov_node_size_real[filler_lhs:filler_rhs]
    else:
        new_filler_area_total = last_filler_area_total

    # pin rel cpos should be scaled by real inflate_ratio
    inflate_ratio = new_mov_node_size_real / ori_mov_node_size_real
    new_pin_rel_cpos = routeforce.inflate_pin_rel_cpos(
        inflate_ratio,
        route_cache.original_pin_rel_cpos,
        data.pin_id2node_id,
        mov_rhs - filler_lhs
    )

    mov_node_size_real.copy_(new_mov_node_size_real)
    data.pin_rel_cpos.copy_(new_pin_rel_cpos)
    old_target_density = copy.deepcopy(args.target_density)
    args.target_density = (new_mov_area_total + new_filler_area_total) / route_cache.placeable_area
    logger.info("Update target density from %.4f to %.4f" % (old_target_density, args.target_density))

    if decrease_target_density:
        # since we decrease target density, update the init density map correspondingly
        logger.warning("Remove nodes to reduce target density from %.4f to %.4f. This step may remove some FloatMov. #Nodes from %d to %d" % 
            (old_target_density, args.target_density, filler_rhs, num_remain_cells))
        data.init_density_map.clamp_(min=0.0, max=1.0).div_(old_target_density).mul_(args.target_density)

    logger.info("Absolute area (last) | mov: %.4E, filler: %.4E, all_cells: %.4E" % (
        last_mov_area_total, last_filler_area_total, last_mov_area_total + last_filler_area_total
    ))
    logger.info("Absolute area (this) | mov: %.4E, filler: %.4E, all_cells: %.4E" % (
        new_mov_area_total, new_filler_area_total, new_mov_area_total + new_filler_area_total
    ))
    logger.info("Relative area change | increment: %.4f, mov: %.4f, filler: %.4f, all_cells: %.4f" % (
        inc_mov_area_total / last_mov_area_total,
        new_mov_area_total / last_mov_area_total,
        new_filler_area_total / last_filler_area_total if last_filler_area_total > 1e-5 else 0,
        (new_mov_area_total + new_filler_area_total) / (last_mov_area_total + last_filler_area_total)
    ))
    logger.info("Inflation Rate | movable: avgX/maxX %.4f/%.4f avgY/maxY %.4f/%.4f, filler: %.4f" % (
        inflate_ratio[mov_lhs:filler_lhs, 0].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 0].max().item(),
        inflate_ratio[mov_lhs:filler_lhs, 1].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 1].max().item(),
        filler_scale
    ))

    if abs(args.target_density * route_cache.placeable_area - torch.sum(torch.prod(mov_node_size_real, 1)).item()) > 1:
        logger.warning("Please check inflation...")
        logger.info("new_total_mov_area: %.1f ori_total_mov_area: %.1f placeable_area: %.1f target_density: %.2f placeable_area * target_density: %.1f" % (
            torch.sum(torch.prod(mov_node_size_real, 1)).item(),
            torch.sum(torch.prod(route_cache.original_mov_node_size, 1)).item(),
            route_cache.placeable_area,
            args.target_density,
            args.target_density * route_cache.placeable_area
        ))
    
    route_cache.mov_node_size_real = mov_node_size_real
    data.mov_node_area = torch.prod(mov_node_size_real, 1).unsqueeze_(1)

    mov_node_size = mov_node_size_real
    expand_ratio = mov_node_size.new_ones((mov_node_size.shape[0]))
    if args.clamp_node:
        mov_node_area = torch.prod(mov_node_size, 1)
        clamp_mov_node_size = mov_node_size.clamp(min=data.unit_len * math.sqrt(2))
        clamp_mov_node_area = torch.prod(clamp_mov_node_size, 1)
        # update
        expand_ratio = mov_node_area / clamp_mov_node_area
        mov_node_size = clamp_mov_node_size

    if decrease_target_density:
        # FIXME: should we update total_mov_area_without_filler when decrease_target_density == False?
        mov_cell_area = torch.prod(new_mov_node_size_real[mov_lhs:mov_rhs, ...], 1)
        data.__total_mov_area_without_filler__ = torch.sum(mov_cell_area).item()

    ps.use_cell_inflate = True
    return gr_metrics, mov_node_size, expand_ratio

def initialize_route_cache(route_cache, data, args, mov_node_size, mov_lhs, filler_lhs, filler_rhs):
    # TODO: check args.target_density should be changed or not?
    route_cache.original_mov_node_size = mov_node_size.clone()
    # NOTE: size_real denotes the node size before node expand
    # these size_real are internally maintained by route_cache 
    route_cache.mov_node_size_real = data.mov_node_size_real.clone()
    route_cache.original_mov_node_size_real = data.mov_node_size_real.clone()
    mov_node_size_real = route_cache.mov_node_size_real

    mov_conn_size = mov_node_size_real[mov_lhs:filler_lhs, ...]
    filler_size = mov_node_size_real[filler_lhs:filler_rhs, ...]

    tmp_area = torch.sum(torch.prod(mov_node_size_real, 1)).item()
    route_cache.target_area = tmp_area
    route_cache.placeable_area = tmp_area / args.target_density
    route_cache.whitespace_area = route_cache.placeable_area - torch.sum(torch.prod(mov_conn_size, 1)).item()
    route_cache.original_num_fillers = filler_rhs - filler_lhs
    route_cache.original_filler_area_total = torch.sum(torch.prod(filler_size, 1)).item()
    route_cache.original_pin_rel_cpos = data.pin_rel_cpos.clone()
    route_cache.original_target_density = copy.deepcopy(args.target_density)
    route_cache.original_total_mov_area_without_filler = copy.deepcopy(data.__total_mov_area_without_filler__)
    route_cache.original_init_density_map = data.init_density_map.clone()
    
    # for momentum-based cell inflation
    route_cache.inflate_ratio_prev = torch.ones_like(mov_conn_size)
    route_cache.delta_inflate_ratio_prev = torch.zeros_like(mov_conn_size)

def cal_cell_inflation_criticality(inflate_ratio_from_cg_map, initialize=False, method=3):
    if initialize:
        return inflate_ratio_from_cg_map - 1.0
    
    # Common variables
    cell_criticality = inflate_ratio_from_cg_map - 1.0
    indices_positives = cell_criticality > 0.0
    mean_value = cell_criticality[indices_positives].mean()
    last_inflate_ratio = route_cache.inflate_ratio_prev.clone()
    last_inflate_ratio_from_cg_map = route_cache.inflate_ratio_from_cg_map_recorder[-2].clone()

    # Method 1: return cell_criticality as is
    def method_1():
        return cell_criticality  
    
    # Method 2: subtract the mean value from cell_criticality where it's greater than 0.0
    def method_2():
        cell_criticality[indices_positives] -= mean_value
        return cell_criticality  

    def method_3():
        indices_last_high_congestion = last_inflate_ratio_from_cg_map > last_inflate_ratio_from_cg_map.mean()
        indices_this_low_congestion = inflate_ratio_from_cg_map < inflate_ratio_from_cg_map.mean()
        indices = indices_last_high_congestion & indices_this_low_congestion
        cell_criticality[indices] = -0.8 * (last_inflate_ratio[indices] - 1.0)
        return cell_criticality 

    # Method 4: update cell_criticality based on specific conditions
    def method_4():
        indices_last_high_congestion = last_inflate_ratio_from_cg_map > last_inflate_ratio_from_cg_map.mean()
        indices_this_low_congestion = inflate_ratio_from_cg_map < inflate_ratio_from_cg_map.mean()
        indices = indices_last_high_congestion & indices_this_low_congestion
        cell_criticality[indices] = -1 * (last_inflate_ratio[indices] - 1.0)
        return cell_criticality
    
    # Method 5: placeholder for additional method
    def method_5():
        return cell_criticality  # Fill in the code for method 5 here

    # Create a dictionary that maps method numbers to their corresponding functions
    methods = {1: method_1, 2: method_2, 3: method_3, 4: method_4, 5: method_5}

    # Use the get method of the dictionary to get the function corresponding to the method number and call it
    return methods.get(method, method_1)()
    
def route_inflation_roll_back(args, logger, data, mov_node_size):
    if not route_cache.first_run:
        mov_lhs, mov_rhs = data.movable_index
        mov_node_size[mov_lhs:mov_rhs].copy_(route_cache.original_mov_node_size[mov_lhs:mov_rhs])
        data.pin_rel_cpos.copy_(route_cache.original_pin_rel_cpos)
        data.__total_mov_area_without_filler__ = route_cache.original_total_mov_area_without_filler
        if args.target_density != route_cache.original_target_density:
            args.target_density = route_cache.original_target_density
            data.init_density_map.copy_(route_cache.original_init_density_map)
    route_cache.reset()

def calc_node_congestion_value(cg_mapAll, node_pos, num_bin_x, num_bin_y, unit_len_x, unit_len_y):
    # Convert all node positions to integer coordinates
    node_pos_x = (node_pos[:, 0] / unit_len_x).long()
    node_pos_y = (node_pos[:, 1] / unit_len_y).long()

    # Clamp all coordinates within valid range
    node_pos_x = torch.clamp(node_pos_x, 0, num_bin_x - 1)
    node_pos_y = torch.clamp(node_pos_y, 0, num_bin_y - 1)

    # Retrieve congestion values for all nodes at once
    node_cg_value = cg_mapAll[node_pos_x, node_pos_y]

    return node_cg_value

def set_macro_region_congestion_one(cg_mapAll, data, routeforce):
    num_bin_x, num_bin_y = cg_mapAll.shape[0], cg_mapAll.shape[1]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width
    lhs, rhs = data.node_type_indices[2][0], data.node_type_indices[2][1]
    node_pos = data.node_pos[lhs:rhs]
    node_size = data.node_size[lhs:rhs]

    node_region_min = node_pos - 0.5  * node_size
    node_region_max = node_pos + 0.5  * node_size

    bin_min_x = (node_region_min[:, 0] / unit_len_x).int().clamp(0, num_bin_x - 1)
    bin_max_x = (node_region_max[:, 0] / unit_len_x).int().clamp(0, num_bin_x - 1)
    bin_min_y = (node_region_min[:, 1] / unit_len_y).int().clamp(0, num_bin_y - 1)
    bin_max_y = (node_region_max[:, 1] / unit_len_y).int().clamp(0, num_bin_y - 1)
    
    
    for i in range(0, rhs-lhs):
        cg_mapAll[bin_min_x[i]:bin_max_x[i], bin_min_y[i]:bin_max_y[i]] = 1
    

def classify_cell_according_pins(mov_node_to_num_pins, lhs, rhs, std_top_multiplier=0.5, std_low_multiplier=0.5):
    # Check if the result is already in the cache
    #key = (lhs, rhs, std_top_multiplier, std_low_multiplier)
    #if key in route_cache.classfied_cell_indices:
    #    return route_cache.classfied_cell_indices[key]

    # If the result is not in the cache, calculate it and save it to the cache
    cell_to_num_pins = mov_node_to_num_pins[lhs:rhs]
    mean = torch.mean(cell_to_num_pins)
    std = torch.std(cell_to_num_pins)

    num_pins_high = mean + std_top_multiplier * std
    num_pins_low = mean - std_low_multiplier * std

    high_indices = torch.nonzero(cell_to_num_pins > num_pins_high, as_tuple=True)[0] + lhs
    medium_indices = torch.nonzero((cell_to_num_pins > num_pins_low) & (cell_to_num_pins <= num_pins_high), as_tuple=True)[0] + lhs
    low_indices = torch.nonzero(cell_to_num_pins <= num_pins_low, as_tuple=True)[0] + lhs

    result = high_indices, medium_indices, low_indices
    #route_cache.classfied_cell_indices[key] = result

    return result

def handle_macro_margin_init_density(cg_mapAll, data, args, routeforce, margin=2):  
    if data.use_whitespace_redistribution:
        init_density = data.init_density_map.clone()
        num_bin_x, num_bin_y = data.num_bin_x, data.num_bin_y
        # assert num_bin_x == cg_mapAll.shape[0] and num_bin_y == cg_mapAll.shape[1]
        #FIXME: the size of cg_mapAll is not always equal to num_bin_x and num_bin_y
        assert abs(num_bin_x - cg_mapAll.shape[0]) <=2 and abs(num_bin_y - cg_mapAll.shape[1]) <=2 

        unit_len_x, unit_len_y = data.unit_len[0], data.unit_len[1]
        macro_pos = data.node_pos[data.node_type_indices[2][0]:data.node_type_indices[2][1]]
        macro_size = data.node_size[data.node_type_indices[2][0]:data.node_type_indices[2][1]]
        
        x_start = torch.clamp(((macro_pos[:, 0] - macro_size[:, 0]/2) / unit_len_x).floor().int(), 0, num_bin_x - 1)
        x_end = torch.clamp(((macro_pos[:, 0] + macro_size[:, 0]/2) / unit_len_x).floor().int(), 0, num_bin_x - 1)
        y_start = torch.clamp(((macro_pos[:, 1] - macro_size[:, 1]/2) / unit_len_y).floor().int(), 0, num_bin_y - 1)
        y_end = torch.clamp(((macro_pos[:, 1] + macro_size[:, 1]/2) / unit_len_y).floor().int(), 0, num_bin_y - 1)
        
        x_start_margin = torch.clamp(x_start - margin, 0, num_bin_x - 1)
        x_end_margin = torch.clamp(x_end + margin, 0, num_bin_x - 1)
        y_start_margin = torch.clamp(y_start - margin, 0, num_bin_y - 1)
        y_end_margin = torch.clamp(y_end + margin, 0, num_bin_y - 1)
        
        for i in range(macro_pos.shape[0]):
            #  only set init_density to args.target_density when cg_mapAll > 1.0
            min_allow_cg = torch.tensor(0.9, dtype=torch.float).to(cg_mapAll.device)
            marco_around_density = torch.tensor(args.target_density, dtype=torch.float).to(cg_mapAll.device)
            data.init_density_map[x_start_margin[i]:x_start[i], y_start_margin[i]:y_end_margin[i]] = torch.where(
                cg_mapAll[x_start_margin[i]:x_start[i], y_start_margin[i]:y_end_margin[i]] > min_allow_cg, marco_around_density, 
                data.init_density_map[x_start_margin[i]:x_start[i], y_start_margin[i]:y_end_margin[i]])
            
            data.init_density_map[x_end[i]:x_end_margin[i], y_start_margin[i]:y_end_margin[i]] = torch.where(
                cg_mapAll[x_end[i]:x_end_margin[i], y_start_margin[i]:y_end_margin[i]] > min_allow_cg, marco_around_density, 
                data.init_density_map[x_end[i]:x_end_margin[i], y_start_margin[i]:y_end_margin[i]])
            
            data.init_density_map[x_start[i]:x_end[i], y_start_margin[i]:y_start[i]] = torch.where(
                cg_mapAll[x_start[i]:x_end[i], y_start_margin[i]:y_start[i]] > min_allow_cg, marco_around_density, 
                data.init_density_map[x_start[i]:x_end[i], y_start_margin[i]:y_start[i]])
            
            data.init_density_map[x_start[i]:x_end[i], y_end[i]:y_end_margin[i]] = torch.where(
                cg_mapAll[x_start[i]:x_end[i], y_end[i]:y_end_margin[i]] > min_allow_cg, marco_around_density, 
                data.init_density_map[x_start[i]:x_end[i], y_end[i]:y_end_margin[i]])
