import torch
from .param_scheduler import ParamScheduler
from .core import merged_wl_loss_grad, WAWirelengthLoss, WAWirelengthLossAndHPWL


def calc_loss(wl_loss, density_loss, ps, args):
    if args.loss_type == "weighted_sum":
        loss = (wl_loss + ps.density_weight * density_loss) / (1 + ps.density_weight)
    elif args.loss_type == "direct":
        loss = wl_loss + ps.density_weight * density_loss
    else:
        raise NotImplementedError("Loss type not defined")
    return loss


def apply_precond(mov_node_pos: torch.Tensor, ps: ParamScheduler, args):
    if not args.use_precond:
        return
    mov_node_pos.grad /= ps.precond_weight
    return mov_node_pos.grad


def calc_obj_and_grad(
    mov_node_pos,
    constraint_fn=None,
    route_fn=None,
    mov_node_size=None,
    expand_ratio=None,
    init_density_map=None,
    density_map_layer=None,
    conn_fix_node_pos=None,
    ps=None,
    data=None,
    args=None,
    merged_forward_backward=True,
):
    logger = ps.__logger__
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos = constraint_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat([conn_node_pos, conn_fix_node_pos], dim=0)
    wire_force, density_force, mov_route_force, mov_congest_force, mov_pseudo_force, all_route_force = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if merged_forward_backward:
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()

        if ps.use_route_force and ps.open_route_force_opt:
            mov_route_grad, mov_congest_grad, mov_pseudo_grad = route_fn(
                mov_node_pos, mov_node_size, expand_ratio, constraint_fn
            )
            mov_node_pos.grad[mov_lhs:mov_rhs] += ps.route_weight * mov_route_grad[mov_lhs:mov_rhs]
            mov_route_force = ps.route_weight * mov_route_grad[mov_lhs:mov_rhs].detach().norm(p=1)
            mov_node_pos.grad += ps.congest_weight * mov_congest_grad
            mov_congest_force = ps.congest_weight * mov_congest_grad.detach().norm(p=1)
            if mov_pseudo_grad is not None:
                mov_node_pos.grad += ps.pseudo_weight * mov_pseudo_grad
                mov_pseudo_force = ps.pseudo_weight * mov_pseudo_grad.detach().norm(p=1)
                ps.grad_recorder["mov_pseudo_grad"] = mov_pseudo_grad.clone()
                
            all_route_force = mov_route_force + mov_congest_force + mov_pseudo_force
            ps.grad_recorder["all_route_grad"] = mov_node_pos.grad.clone()
            ps.grad_recorder["mov_route_grad"] = mov_route_grad.clone()
            ps.grad_recorder["mov_congest_grad"] = mov_congest_grad.clone()
            
        wl_loss, conn_node_grad_by_wl = merged_wl_loss_grad(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
            data.hpwl_scale, ps.wa_coeff, args.deterministic
        )
        ps.grad_recorder["conn_node_grad_by_wl"] = conn_node_grad_by_wl.clone()
        mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_wl[mov_lhs:mov_rhs]
        wire_force = conn_node_grad_by_wl[mov_lhs:mov_rhs].detach().norm(p=1)
        
        if ps.enable_sample_force:
            if ps.iter > 3 and ps.iter % 20 == 0:
                # ps.iter > 3 for warmup
                density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                    mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
                )
                ps.grad_recorder["node_grad_by_density"] = node_grad_by_density.clone()
                mov_node_pos.grad += ps.density_weight * node_grad_by_density
                density_force = ps.density_weight * node_grad_by_density.detach().norm(p=1)
            else:
                density_loss = 0.0
            if (ps.iter > 3 and ps.recorder.density_force_ratio[-1] > 1e-2) or ps.iter > 100:
                # no longer enable sampling back
                ps.enable_sample_force = False
        else:
            density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
            )
            ps.grad_recorder["node_grad_by_density"] = node_grad_by_density.clone()
            density_force = ps.density_weight * node_grad_by_density.detach().norm(p=1)
            if ps.use_route_force and ps.open_route_force_opt and ps.grad_recorder["all_route_grad"] is not None:
                non_zero_row_indices = torch.nonzero(ps.grad_recorder["all_route_grad"].any(dim=1), as_tuple=True)[0]
                node_grad_by_density[non_zero_row_indices, :] = 0
            mov_node_pos.grad += ps.density_weight * node_grad_by_density
            
        ps.density_force_ratio = (density_force / wire_force).clamp_(max=30)
        ps.mov_route_force_ratio = (mov_route_force / wire_force).clamp_(max=10)
        ps.mov_congest_force_ratio = (mov_congest_force / wire_force).clamp_(max=10)
        ps.mov_pseudo_force_ratio = (mov_pseudo_force / wire_force).clamp_(max=10)
        ps.all_route_force_ratio = (all_route_force / wire_force).clamp_(max=10)
        
        grad = apply_precond(mov_node_pos, ps, args)
        loss = wl_loss + ps.density_weight * density_loss
        
    else:#TODO: add route force
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()
        wl_loss = WAWirelengthLoss.apply(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
            ps.wa_coeff, args.deterministic
        )
        density_loss, _ = density_map_layer(
            mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
        )
        loss = calc_loss(wl_loss, density_loss, ps, args)
        loss.backward()
        grad = apply_precond(mov_node_pos, ps, args)
    return loss, grad


def calc_grad(
    optimizer: torch.optim.Optimizer, mov_node_pos: torch.Tensor, wl_loss, density_loss
):
    optimizer.zero_grad(set_to_none=False)
    wl_loss.backward(retain_graph=True)
    wl_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad(set_to_none=False)
    density_loss.backward(retain_graph=True)
    density_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad(set_to_none=False)
    return wl_grad, density_grad


def fast_optimization(
    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
    density_map_layer, mov_node_size, init_density_map, ps, data, args
):  
    mov_node_pos = trunc_node_pos_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat(
        [conn_node_pos, conn_fix_node_pos], dim=0
    )
    wl_loss, hpwl = WAWirelengthLossAndHPWL.apply(
        conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
        data.node2pin_list, data.node2pin_list_end,
        data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
        ps.wa_coeff, data.hpwl_scale, args.deterministic
    )
    density_loss, overflow = density_map_layer(
        mov_node_pos, mov_node_size, init_density_map
    )
    loss = calc_loss(wl_loss, density_loss, ps, args)
    loss.backward()
    apply_precond(mov_node_pos, ps, args)
    # calculate objective (hpwl, overflow)
    return hpwl.detach(), overflow.detach(), mov_node_pos