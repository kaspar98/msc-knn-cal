import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .temp_scaling import softmax
from .proper_losses import bs, log_loss

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_cf_triangle(true_cal_fun, cal_fun, title, data_train, data_test, directory):
    temp = 1
    print(title)
    bscore = f'BS={np.round(bs(cal_fun(data_train["p"]), data_train["y"]), 8)}'
    ll = f'LL={np.round(log_loss(y_true=data_train["y"], y_pred=cal_fun(data_train["p"])), 8)}'
    ce = np.mean(np.abs(true_cal_fun(data_train["p"]) - cal_fun(data_train["p"])))

    print(f"CE train {ce}")
    print(f"BS train {bscore}")
    print(f"LL train {ll}")
    ll = f'LL={np.round(log_loss(y_true=data_test["y"], y_pred=cal_fun(data_test["p"])), 8)}'
    ce = np.mean(np.abs(true_cal_fun(data_test["p"]) - cal_fun(data_test["p"])))
    print(f"CE test {ce}")
    print(f"BS test {bscore}")
    print(f"LL test {ll}")

    fig = set_up_3d_fig(subplots=1)
    ax1, axis_idx = add_triangle_axis(fig, 1, 1, temp_param=temp, step=0.2)
    plot_ax_to_triangle_guidelines(ax1, step=0.2)
    plot_on_triangle_guidelines(ax1, thick_borders=False, temp_param=temp, step=0.2)

    # Arrows
    arrow_starts = generate_points_on_triangle(step_size=(1 - 0.02375 * 3) / 20, rounding_to=7,
                                                   start=0.02375)
    arrow_starts_scaled = temp_scale_points(arrow_starts, temp)
    arrow_ends = cal_fun(arrow_starts_scaled)
    for idx in range(len(arrow_starts)):
        draw_classic_arrow(arrow_starts[idx], arrow_ends[idx], ax=ax1, alpha=0.5)

    points = generate_points_on_triangle()
    cal_points_scaled = temp_scale_points(points, temp)
    cal_points = cal_fun(cal_points_scaled)
    CE_estimates = np.clip(((points - cal_points) * 6 + 0.5), a_min=0,
                           a_max=1)  # np.clip([1,1,1] - (mean_p_s - mean_y_s) * 10, a_min=0, a_max=1)
    sc = scatter_color_points_on_triangle(ax1, points=points, colors=CE_estimates, use_log_colors=False)

    plt.title(title + f"\n$CE^1={np.round(ce,4)}$"+f"\n{ll}"+f"\n{bscore}")
    plt.savefig(directory + "/simplex_" + title + ".jpg", dpi=250, bbox_inches="tight")
    plt.show()


def draw_classic_arrow(arrow_start, arrow_end, ax, alpha=1, color="black", lw=1, zorder=20, arrowstyle="->", mutation_scale=12, shrinkA=0, shrinkB=0, **kwargs):
    arw = Arrow3D([arrow_start[0], arrow_end[0]],
                  [arrow_start[1], arrow_end[1]],
                  [arrow_start[2], arrow_end[2]],
                  arrowstyle=arrowstyle, color=color, lw=lw, mutation_scale=mutation_scale, zorder=zorder, alpha=alpha,
                  shrinkA=shrinkA, shrinkB=shrinkB, **kwargs)
    ax.add_artist(arw)


def un_temp_scale_points(p, temp_param=10):
    p = np.array(p)
    p[p == 0] = 1e-10

    return softmax(np.log(p) * temp_param)


def temp_scale_points(p, temp_param):
    p = np.clip(p, a_min=0, a_max=1)
    p[p == 0] = 1e-10
    return un_temp_scale_points(p, temp_param=1 / temp_param)


def generate_points_on_triangle(step_size=0.005, rounding_to=4, start=0.0):
    xx, yy = np.meshgrid(np.arange(start, 1 + step_size / 2, step_size), np.arange(start, 1 + step_size / 2, step_size))
    xx, yy = xx[xx + yy <= 1], yy[xx + yy <= 1]
    xx = np.round(xx, rounding_to)
    yy = np.round(yy, rounding_to)
    z = 1 - xx - yy
    z = np.round(z, rounding_to)

    p = np.dstack((xx, yy, z))[0]
    p = p[(p[:, 2] >= start)]

    return p


def scatter_color_points_on_triangle(ax, points, colors, zorder=10, use_log_colors=False, alpha=1):
    cmap = copy.copy(plt.get_cmap('YlOrRd'))

    if use_log_colors:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name="new_cmap", colors=cmap(np.linspace(0.2, 1.0, 6)))
        bounds = [1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        norm = None

    points = np.array(points)
    colors = np.array(colors)

    if len(colors.shape) == 1:
        colors = np.ma.masked_where(colors == 0, colors)

    scatter_points = ax.scatter(points[:, 0],
                                points[:, 1],
                                points[:, 2],
                                alpha=alpha, c=colors, s=1.5, zorder=zorder, cmap=cmap, norm=norm)

    return scatter_points


def scatter_points_and_print_untemp_scaled(p, temp):
    fig = set_up_3d_fig(subplots=1, figheight=12)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)

    ax = add_triangle_axis(fig, 1, 1, temp_param=temp, step=0.1)

    plot_ax_to_triangle_guidelines(ax)
    plot_on_triangle_guidelines(ax=ax, thick_borders=True, temp_param=temp, step=0.1)

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], alpha=1, c="red", s=15)
    plt.show()

    print("Scaled")
    print(un_temp_scale_points(p, temp))


def plot_temp_scale_sample_points_with_text(ax, temp_param):
    p = generate_points_on_triangle(step_size=0.85 / 5, rounding_to=6, start=0.05)
    scaled_p = un_temp_scale_points(p, temp_param)
    for idx in range(len(p)):
        ax.text(p[idx, 0], p[idx, 1], p[idx, 2], f"{np.round(scaled_p[idx], 4)}", size=11, zorder=40, color='k')
    ax.scatter(p[:, 0] + 0.01, p[:, 1] + 0.01, p[:, 2] + 0.01, color='red', zorder=50, s=10, alpha=1)


def plot_ax_to_triangle_guidelines(ax, c="black", zorder=20, step=0.2):
    alpha = 0.5
    lw = 0.5

    for i in np.arange(0, 1, step):
        ax.plot([i, i], [1, 1 - i], [0, 0], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([1, 1 - i], [i, i], [0, 0], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([1, 1 - i], [0, 0], [i, i], color=c, alpha=alpha, lw=lw, zorder=zorder)


def plot_classic_on_triangle_guidelines(ax, c="black", zorder=20, step=0.2, darker=False):
    alpha = 0.5
    lw = 0.5

    for i in np.arange(0, 1, step):

        if darker:
            if i % (step * 5) == 0:
                alpha = 0.8
                lw = 1.5
            else:
                alpha = 0.5
                lw = 0.5

        ax.plot([0, 1 - i], [1 - i, 0], [i, i], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([i, i], [1 - i, 0], [0, 1 - i], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([0, 1 - i], [i, i], [1 - i, 0], color=c, alpha=alpha, lw=lw, zorder=zorder)


def plot_approx_on_triangle_guidelines(ax, temp_param, zorder, c, step=0.2):
    alpha = 0.5
    lw = 0.5
    i = 0
    ax.plot([0, 1 - i], [1 - i, 0], [i, i], color=c, alpha=alpha, lw=lw, zorder=zorder)
    ax.plot([i, i], [1 - i, 0], [0, 1 - i], color=c, alpha=alpha, lw=lw, zorder=zorder)
    ax.plot([0, 1 - i], [i, i], [1 - i, 0], color=c, alpha=alpha, lw=lw, zorder=zorder)

    helper_lines_at = np.arange(step, 1, step)

    for dimens in [0, 1, 2]:
        for helper_idx, helper_line in enumerate(helper_lines_at):
            scaled_helper_at = un_temp_scale_points([[helper_line, 0.0, 1 - helper_line]], temp_param=temp_param)[0][0]

            coords1 = np.arange(0, (1 - scaled_helper_at) * 1.025, (1 - scaled_helper_at) * 0.05)
            coords2 = 1 - scaled_helper_at - coords1
            coords = np.zeros((len(coords1), 3))
            coords[:, dimens] = scaled_helper_at
            coords[:, (dimens + 1) % 3] = coords1
            coords[:, (dimens + 2) % 3] = coords2

            scaled_coords = temp_scale_points(coords, temp_param)
            ax.plot(scaled_coords[:, 0],
                    scaled_coords[:, 1],
                    scaled_coords[:, 2], "--", color=c, zorder=zorder, alpha=0.5, lw=0.7)


def plot_on_triangle_guidelines(ax, c="black", zorder=20, thick_borders=True, temp_param=1, step=0.2):
    if temp_param == 1:
        plot_classic_on_triangle_guidelines(ax=ax, c=c, zorder=zorder, step=step)
    else:
        plot_approx_on_triangle_guidelines(ax=ax, temp_param=temp_param, zorder=zorder, c=c, step=step)

    if thick_borders:
        alpha = 1.0
        lw = 3
        i = 0
        ax.plot([0, 1 - i], [1 - i, 0], [i, i], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([i, i], [1 - i, 0], [0, 1 - i], color=c, alpha=alpha, lw=lw, zorder=zorder)
        ax.plot([0, 1 - i], [i, i], [1 - i, 0], color=c, alpha=alpha, lw=lw, zorder=zorder)


def add_triangle_axis(fig, axis_idx, n_axes, temp_param=1, step=0.2):
    ax = fig.add_subplot(1, n_axes, axis_idx, projection='3d')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_xlabel("$\hat{p}_1$")
    ax.set_ylabel("$\hat{p}_2$")
    ax.set_zlabel("$\hat{p}_3$")

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10

    ax.view_init(elev=20, azim=45) # elev=30
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)

    ticks = np.arange(0.0, 1.01, step)
    coords = [[coord, 0, 1 - coord] for coord in ticks]
    scaled_ticks = un_temp_scale_points(coords, temp_param)[:, 0]

    ax.set_xticks(ticks)
    ax.set_xticklabels(np.round(scaled_ticks, 4))

    ax.set_yticks(ticks)
    ax.set_yticklabels(np.round(scaled_ticks, 4))

    ax.set_zticks(ticks)
    ax.set_zticklabels(np.round(scaled_ticks, 4))

    # ax.patch.set_linewidth(3) # ääred subplotile
    # ax.patch.set_edgecolor('black')

    return ax, axis_idx + 1


def set_up_3d_fig(figheight=8, subplots=1):
    figwidth = 1 * figheight * subplots  # 2.2
    figsize = (figwidth, figheight)
    m = 20
    s = 12
    plt.rc("axes", titlesize=m)  # fontsize of the axes title
    plt.rc("axes", labelsize=s)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=s)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=s)

    fig = plt.figure(figsize=figsize)

    # fig.patch.set_linewidth(10) # ääred
    # fig.patch.set_edgecolor('cornflowerblue')

    return fig