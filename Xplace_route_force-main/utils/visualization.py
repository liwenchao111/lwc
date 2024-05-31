from typing import List, Tuple
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import imageio.v2 as imageio
from glob import glob

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.INFO)


def scatter_drawer(pos: torch.Tensor, fix_mask: torch.Tensor, filename, title, args):
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    # pos = pos.cpu().numpy()
    # pos = pos.T
    # plt.scatter(pos[0], pos[1])
    mov_pos = pos[fix_mask.squeeze(1) < 0.5].T.cpu().numpy()
    fix_pos = pos[fix_mask.squeeze(1) > 0.5].T.cpu().numpy()
    plt.scatter(mov_pos[0], mov_pos[1], label="mov")
    plt.scatter(fix_pos[0], fix_pos[1], label="fix")
    plt.legend()
    plt.title(title)
    plt.savefig(png_path)
    plt.close()


def draw_fig(batch, pos, fix_mask, info, args):
    epoch, idx, iteration, hpwl = info
    filename = "epoch%d_id%d_iter%d.png" % (epoch, idx, iteration)
    title = "hpwl %.4f" % hpwl
    num_items = batch.num_of_graph_nodes[0]
    scatter_drawer(pos[:num_items], fix_mask[:num_items], filename, title, args)


def scatter_drawer_new(pos: torch.Tensor, filename, title, args):
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    pos = pos.cpu().numpy()
    pos = pos.T
    plt.scatter(pos[0], pos[1])
    plt.title(title)
    plt.savefig(png_path)
    plt.close()


def draw_fig_new(pos, info, args):
    iteration, hpwl, design_name = info
    filename = "%s_iter%d.png" % (design_name, iteration)
    title = "hpwl %.4f" % hpwl
    scatter_drawer_new(pos, filename, title, args)

def draw_nodes(ctx, node_pos, node_size, color, lx, ly):
    if node_pos is not None:
        node_pos = node_pos.cpu()
        node_size = node_size.cpu()

        for i in range(node_pos.shape[0]):
            pos_x = (node_pos[i][0].item() + lx)
            pos_y = (node_pos[i][1].item() + ly)
            size_x = (node_size[i][0].item())
            size_y = (node_size[i][1].item())

            ctx.rectangle(pos_x - size_x / 2, pos_y - size_y / 2, size_x, size_y)
            ctx.set_source_rgba(*color)
            ctx.fill()
        
def draw_grid(ctx, lx, hx, ly, hy, num_bin_x, num_bin_y):
    lineWidth = 0.05 * min((hx - lx) / num_bin_x, (hy - ly) / num_bin_y)
    ctx.set_line_width(lineWidth)
    ctx.set_source_rgb(0.3, 0.3, 0.3)

    for i in range(1, num_bin_x):
        ctx.move_to(i * (hx - lx) / num_bin_x + lx, ly)
        ctx.line_to(i * (hx - lx) / num_bin_x + lx, hy)
        ctx.stroke()

    for i in range(1, num_bin_y):
        ctx.move_to(lx, i * (hy - ly) / num_bin_y + ly)
        ctx.line_to(hx, i * (hy - ly) / num_bin_y + ly)
        ctx.stroke()  

def draw_arrow(ctx, x1, y1, x2, y2, color, arrow_size=10):
    # Calculate the direction of the arrow
    angle = math.atan2(y2 - y1, x2 - x1) + math.pi

    # Draw the body of the arrow
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.set_line_width(0.3 * arrow_size)  # Set the line width to 2
    ctx.stroke()

    # Draw the head of the arrow
    ctx.move_to(x2, y2)
    ctx.line_to(x2 + arrow_size * math.cos(angle - 0.5), y2 + arrow_size * math.sin(angle - 0.5))
    ctx.line_to(x2 + arrow_size * math.cos(angle + 0.5), y2 + arrow_size * math.sin(angle + 0.5))
    ctx.close_path()

    ctx.set_source_rgba(*color)
    ctx.fill()

def draw_route_force_arrows(ctx, mov_node_pos, route_grad, scale_factor, topk, color, lx, ly):
    if topk > route_grad.shape[0]:
        topk = route_grad.shape[0]
    
    route_grad = route_grad.cpu()
    norms = torch.norm(route_grad, p=1, dim=1)
    _, topk_indices = torch.topk(norms, topk)

    max_norm_index = topk_indices[0]
    max_norm = torch.norm(route_grad[max_norm_index], p=2)
    normalized_grad = route_grad / max_norm * scale_factor

    for i in topk_indices:
        begin_x = round(mov_node_pos[i][0].item() + lx)
        begin_y = round(mov_node_pos[i][1].item() + ly)
        end_x = begin_x - normalized_grad[i][0].item()
        end_y = begin_y - normalized_grad[i][1].item()

        draw_arrow(ctx, begin_x, begin_y, end_x, end_y, color, 0.2 * scale_factor)

def draw_fig_with_cairo(
    mov_node_pos,
    mov_node_size,
    fix_node_pos,
    fix_node_size,
    filler_node_pos,
    filler_node_size,
    data,
    info,
    args,
    route_grad=None,
    base_size=2048,
):
    import cairocffi as cairo

    iteration, hpwl, design_name = info
    filename = "%s_iter%d.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
        
    lx, hx, ly, hy = data.die_info.cpu().numpy()
    WIDTH = base_size
    HEIGHT = int(WIDTH * (hx - lx) / (hy - ly))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    # Scale Image
    ratio0, ratio1 = WIDTH / (hx - lx), HEIGHT / (hy - ly)
    ctx.translate(-lx * ratio0, HEIGHT + ly * ratio1)
    ctx.scale(ratio0, -ratio1)
    # White Background
    ctx.rectangle(lx, ly, hx - lx, hy - ly)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.fill()
    # Bins / Grids
    # draw_grid(ctx, lx, hx, ly, hy, data.num_bin_x, data.num_bin_y)
    
    # Nodes
    draw_nodes(ctx, mov_node_pos, mov_node_size, (0.0, 0.0, 0.5, 0.8), lx, ly)
    draw_nodes(ctx, fix_node_pos, fix_node_size, (1.0, 0.0, 0.0, 0.65), lx, ly)
    draw_nodes(ctx, filler_node_pos, filler_node_size, (0.5, 0.5, 0.5, 0.8), lx, ly)

    # Grad Arrows (the biggest k arrows in terms of L1 norm)
    if route_grad is not None and torch.any(route_grad != 0):
        scale_factor = 0.035 * min(hx - lx, hy - ly)
        color = (0.0, 0.0, 0.0, 0.7)
        if filler_node_pos is not None:
            mov_node_pos = torch.cat([mov_node_pos, filler_node_pos], dim=0)
        draw_route_force_arrows(ctx, mov_node_pos, route_grad, scale_factor, 400, color, lx, ly)

    surface.write_to_png(png_path)
    surface.finish()

def draw_fig_with_cairo_cpp(node_pos, node_size, data, info, args, base_size=2048):
    from cpp_to_py import draw_placement

    die_info = tuple(data.__ori_die_info__.tolist())
    scaleX, scaleY = data.die_scale[0].cpu(), data.die_scale[1].cpu()
    shiftX, shiftY = data.die_shift[0].cpu(), data.die_shift[1].cpu()
    lx, hx, ly, hy = die_info

    node_pos_x: List[float] = (node_pos.cpu()[:, 0] * scaleX + shiftX).tolist()
    node_pos_y: List[float] = (node_pos.cpu()[:, 1] * scaleY + shiftY).tolist()
    node_size_x: List[float] = (node_size.cpu()[:, 0] * scaleX).tolist()
    node_size_y: List[float] = (node_size.cpu()[:, 1] * scaleY).tolist()
    node_name: List[str] = ["%d" % i for i in range(node_pos.shape[0])]

    iteration, hpwl, design_name = info
    filename = "%s_iter%d.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path: str = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    site_info = (data.site_width, data.site_height)
    bin_size_info = (
        round(1 / data.num_bin_x * (hx - lx)),
        round(1 / data.num_bin_y * (hy - ly)),
    )
    node_type_indices = data.node_type_indices
    ele_type_to_rgba_vec: List[Tuple[str, float, float, float, float]] = [
        ("Bin", 0.1, 0.1, 0.1, 1.0),
        ("Mov", 0.475, 0.706, 0.718, 0.8),
        ("Filler", 0.8, 0.8, 0.8, 0.8),
    ]
    width = base_size
    height = round(width * (hy - ly) / (hx - lx))
    draw_contents: List[str] = ["Nodes", "NodesText"]

    status = draw_placement.draw(
        node_pos_x,
        node_pos_y,
        node_size_x,
        node_size_y,
        node_name,
        die_info,
        site_info,
        bin_size_info,
        node_type_indices,
        ele_type_to_rgba_vec,
        png_path,
        width,
        height,
        draw_contents,
    )


def visualize_electronic_variables(density_map, potential_map, force_map, info, args):
    import cv2
    iteration, design_name = info
    M, N = density_map.shape

    def get_png_path(filename):
        res_root = os.path.join(args.result_dir, args.exp_id)
        png_path = os.path.join(res_root, args.eval_dir, filename)
        if not os.path.exists(os.path.dirname(png_path)):
            os.makedirs(os.path.dirname(png_path))
        return png_path

    # 1) Visualize density_map
    filename = "%s_iter%d_density.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(density_map.cpu().numpy(), cmap="YlGnBu")
    fig.colorbar(im, ax=ax)
    ax.title.set_text("Density Map")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    # 2) Visualize potential_map
    filename = "%s_iter%d_potential.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(potential_map.cpu().numpy(), cmap="YlGnBu")
    fig.colorbar(im, ax=ax)
    ax.title.set_text("Potential Map")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    # 3) Visualize force_map
    filename = "%s_iter%d_force.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    # 3.1) Init background image
    GRID_SIZE = 100
    img = np.ones((M * GRID_SIZE, N * GRID_SIZE, 3)) * 255

    # 3.2) Draw grid line
    for i in range(0, M * GRID_SIZE - 1, GRID_SIZE):
        cv2.line(img, (i, 0), (i, N * GRID_SIZE), (0, 0, 0), 1, 1)
    for j in range(0, N * GRID_SIZE - 1, GRID_SIZE):
        cv2.line(img, (0, j), (M * GRID_SIZE, j), (0, 0, 0), 1, 1)

    # 3.3) Normalize force
    max_force = torch.sum(torch.pow(force_map, 2), axis=0).sqrt().max().item()
    force_map = (force_map / max_force).cpu().numpy()

    # 3.4) Draw force arrows
    for i in range(0, M, 1):
        centre_x = i * GRID_SIZE + GRID_SIZE / 2
        for j in range(0, N, 1):
            centre_y = j * GRID_SIZE + GRID_SIZE / 2
            cv2.arrowedLine(
                img,
                (
                    int(centre_x - force_map[0][i][j] * GRID_SIZE / 2),
                    int(centre_y - force_map[1][i][j] * GRID_SIZE / 2),
                ),
                (
                    int(centre_x + force_map[0][i][j] * GRID_SIZE / 2),
                    int(centre_y + force_map[1][i][j] * GRID_SIZE / 2),
                ),
                color=(230, 216, 173),
                thickness=10,
                tipLength=0.3,
            )

    cv2.imwrite(png_path, img)


def draw_grad_abs_mean(
    wl_grads, density_grads, iterations, info, args,
):
    iteration, design_name = info
    filename = "%s_iter%d_grad_magnitude_mean.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    colors = ["tab:blue", "tab:red"]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Wirelength Gradient Magnitude", color=colors[0])
    ax1.plot(iterations, wl_grads, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax1.twinx()

    ax2.set_ylabel("Density Gradient Magnitude", color=colors[1])
    ax2.plot(iterations, density_grads, color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    plt.title("Gradient Magnitude Mean")
    fig.tight_layout()
    plt.savefig(png_path)
    plt.close()

def pictures_to_gif(png_path, gif_path):
    png_files = sorted(glob(png_path))
    if len(png_files) == 0:
        print("No png files found in %s" % png_path)
        return
    
    # FIXME: DEBUG (Image.py Line564) Error closing
    # logging.getLogger().setLevel(logging.WARNING) #using this may cause error in Final routing evalution by GGR 
    images = [imageio.imread(str(filename)) for filename in png_files]
    imageio.mimsave(gif_path, images, duration=1)

def generate_gif(args, design_name):
    res_root = os.path.join(args.result_dir, args.exp_id)
    dirs = []
    if args.draw_placement:
        dirs.append(args.eval_dir)
    if args.draw_congestion_map:
        dirs.append(args.route_dir)
        
    for dir in dirs:
        png_path = os.path.join(res_root, dir, "%s_iter*.png" % design_name)
        gif_path = os.path.join(res_root, dir, "%s_%s.gif" % (design_name, dir))
        pictures_to_gif(png_path, gif_path)