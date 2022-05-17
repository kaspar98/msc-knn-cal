import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from sklearn.model_selection import KFold

from .multiclass_CE import CE_estimation_distance
from .plot_helpers import (generate_points_on_triangle,
                                       set_up_3d_fig, add_triangle_axis,
                                       plot_on_triangle_guidelines,
                                       plot_ax_to_triangle_guidelines,
                                       un_temp_scale_points,
                                       scatter_color_points_on_triangle,
                                       temp_scale_points,
                                       draw_classic_arrow,
                                       Arrow3D)

def find_nr_bins_with_cv(p, y, bins_to_try, n_cv_folds, n_trees, equal_size_forest):
    bin_scores = [0] * len(bins_to_try)

    for bin_idx, n_bins in enumerate(bins_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=1)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            forest = equal_size_forest(n_trees=n_trees, n_bins=n_bins, n_classes=p.shape[1])
            forest.fit(p=p_train, y=y_train)

            mean_y_s = forest.mean_y_in_corresponding_leaves(p_test)
            mean_p_s = forest.mean_p_in_corresponding_leaves(p_test)
            CE_estimates = mean_p_s - mean_y_s
            c_hat_test = p_test - CE_estimates
            from kood.helpers.proper_losses import log_loss
            #cv_scores.append(np.mean((c_hat_test - y_test) ** 2))
            cv_scores.append(log_loss(y_test, c_hat_test))

        bin_scores[bin_idx] = np.mean(cv_scores)

    return bin_scores



def find_depth_with_cv(p, y, depths_to_try, n_cv_folds, n_trees, random_split_forest, cv_loss, crop, seed=0):
    depth_scores = [0] * len(depths_to_try)

    for d_idx, depth in enumerate(depths_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=1)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            forest = random_split_forest(n_trees=n_trees, depth=depth, n_classes=p.shape[1])
            forest.fit(p=p_train, y=y_train, random_seed=seed)

            c_hat_test = forest.calibrate_predictions(p_test, crop=crop)

            cv_scores.append(cv_loss(y_test, c_hat_test))

        depth_scores[d_idx] = np.mean(cv_scores)

    return depth_scores

def calibrate_preds_with_random_forest_cv(p_train, y_train, p_test, depths_to_try, n_cv_folds, n_trees, cv_loss, crop, seed=0):

    depth_scores = find_depth_with_cv(p_train, y_train, depths_to_try=depths_to_try, n_cv_folds=n_cv_folds, n_trees=n_trees,
                                  random_split_forest=RandomSplitCalibrationForest, cv_loss=cv_loss, crop=crop, seed=seed)

    print(depth_scores)
    depth_to_use = depths_to_try[np.argmin(depth_scores)]

    forest = RandomSplitCalibrationForest(n_trees=n_trees, depth=depth_to_use, n_classes=y_train.shape[1])
    forest.fit(p_train, y_train, random_seed=seed)
    cal_p_test = forest.calibrate_predictions(p_test, crop=crop)
    cal_p_train = forest.calibrate_predictions(p_train, crop=crop)

    return cal_p_test, cal_p_train, depth_scores

def force_p_to_simplex(p, crop=0):
    # bring the predictions back on the simplex if they move away
    p[p < 0] = crop
    p[p > 1] = 1 - crop
    p = p / np.repeat(np.sum(p, axis=1), p.shape[1]).reshape((-1, p.shape[1]))
    return p

class CalibrationForest:
    def __init__(self, n_trees, n_classes, n_calibrated_copies=0):
        self.n_trees = n_trees
        self.trees = []
        self.n_calibrated_copies = n_calibrated_copies
        self.n_classes = n_classes

    def fit(self, p, y, random_seed=None, class_groups=None):
        """
        class_groups
            None - no grouping of classes
            array - array of three arrays describing the grouping indices
            "random" - random uniform grouping of classes into three equally large groups
            "default" - uniform grouping of classes into three equally large groups by class ordering
        """

        assert self.n_classes == len(p[0])

        if random_seed is not None:
            np.random.seed(random_seed)

        self.n_data = len(p)
        self.original_p = p
        self.original_y = y

        if class_groups is not None:

            if isinstance(class_groups, (list, np.ndarray)):
                g0_idx = class_groups[0]
                g1_idx = class_groups[1]
                g2_idx = class_groups[2]
            elif class_groups == "random":
                g0_idx, g1_idx, g2_idx = np.array_split(np.random.permutation(range(len(p[0]))), 3)
            elif class_groups == "default":
                g0_idx, g1_idx, g2_idx = np.array_split(range(len(p[0])), 3)
            else:
                raise Exception("Bad argument to class_groups")

            p = np.dstack((np.sum(p[:, g0_idx], axis=1), np.sum(p[:, g1_idx], axis=1), np.sum(p[:, g2_idx], axis=1)))[0]
            y = np.dstack((np.sum(y[:, g0_idx], axis=1), np.sum(y[:, g1_idx], axis=1), np.sum(y[:, g2_idx], axis=1)))[0]
            self.n_classes = p.shape[1]

        self.p = p
        self.y = y

        # Calibrated copies

        # Repeat preds for n_calibrated_copies
        repeated_p = np.tile(self.p, (self.n_calibrated_copies, 1))
        # Sample y from preds efficiently
        cumulative_p = repeated_p.cumsum(axis=1)
        choices = (np.random.rand(len(cumulative_p), 1) < cumulative_p).argmax(axis=1)
        # One-hot-encode samples
        repeated_y = np.eye(self.n_classes)[choices]
        # Reshape to n_calibrated_copies
        self.calibrated_copies_y = repeated_y.reshape((self.n_calibrated_copies, self.n_data, self.n_classes))

    def CE_estimate(self, d=2):

        # Should be equal to
        # mean_p_s = np.array([tree.mean_p_in_corresponding_leaves(p) for tree in forest.trees])
        # mean_y_s = np.array([tree.mean_y_in_corresponding_leaves(p) for tree in forest.trees])
        # np.mean(np.power(np.sum(np.power(np.abs(mean_p_s - mean_y_s), d), axis=2), 1/d))
        return np.mean([tree.CE_estimate(d=d) for tree in self.trees])

    def CE_estimates_in_corresponding_leaves(self, p, d=2):
        return np.mean([tree.CE_estimates_in_corresponding_leaves(p, d=d) for tree in self.trees], axis=0)

    def mean_y_in_corresponding_leaves(self, p, use_original_y=False):
        return np.mean([tree.mean_y_in_corresponding_leaves(p, use_original_y=use_original_y) for tree in self.trees],
                       axis=0)

    def mean_calibrated_copies_mean_y_in_corresponding_leaves(self, p):
        # shape (n_trees, n_calibrated_copies, p, n_classes)
        return np.mean([tree.mean_calibrated_copies_mean_y_in_corresponding_leaves(p) for tree in self.trees], axis=0)

    def mean_p_in_corresponding_leaves(self, p, use_original_p=False):
        return np.mean([tree.mean_p_in_corresponding_leaves(p, use_original_p=use_original_p) for tree in self.trees],
                       axis=0)

    def number_of_leaves_in_forest(self):

        if self.n_classes != 3:
            print("Only implemented for 3 classes!")
            return

        p = generate_points_on_triangle()
        CE_estimates_in_leaves = np.mean([tree.CE_estimates_in_corresponding_leaves(p) for tree in self.trees], axis=0)
        return len(np.unique(CE_estimates_in_leaves))

    def calibrate_predictions(self, p_to_calibrate, p_to_calibrate_at=None, crop=1e-6):

        cal_preds = np.zeros(p_to_calibrate.shape)
        for tree in self.trees:
            cal_preds += tree.calibrate_predictions(p_to_calibrate, p_to_calibrate_at=p_to_calibrate_at)
        calibrated_preds = cal_preds / len(self.trees)

        return force_p_to_simplex(calibrated_preds, crop=crop)

    def calibrate_predictions_old(self, p_to_calibrate, p_to_calibrate_at=None, crop=1e-6):
        """
        :param p_to_calibrate: Data values to get calibrated (either grouped according to forest or not grouped and in original shape)
        :param p_to_calibrate_at: Only used if 'p_to_calibrate' is in original shape and doesn't match forest shape. Then this should be. It is used to locate the leaves in trees.
        :return:
        """

        if p_to_calibrate_at is None:
            mean_y_s = self.mean_y_in_corresponding_leaves(p_to_calibrate, use_original_y=False)
            mean_p_s = self.mean_p_in_corresponding_leaves(p_to_calibrate, use_original_p=False)
        else:
            mean_y_s = self.mean_y_in_corresponding_leaves(p_to_calibrate_at, use_original_y=True)
            mean_p_s = self.mean_p_in_corresponding_leaves(p_to_calibrate_at, use_original_p=True)

        CE_estimates = mean_p_s - mean_y_s

        """
        calibrated_preds = p_to_calibrate - CE_estimates
        """
        if self.n_calibrated_copies > 0:
            calibrated_copies_mean_y_s = self.mean_calibrated_copies_mean_y_in_corresponding_leaves(p_to_calibrate)

            CE_estimate_lengths = np.linalg.norm(CE_estimates, axis=1)

            calibrated_copies_CE_estimates = mean_p_s - calibrated_copies_mean_y_s
            calibrated_copies_CE_estimate_lengths = np.linalg.norm(calibrated_copies_CE_estimates, axis=2)
            calibrated_copies_CE_estimate_length_quantiles = np.quantile(calibrated_copies_CE_estimate_lengths, q=1.0,
                                                                         axis=0)

            CE_estimates[CE_estimate_lengths <= calibrated_copies_CE_estimate_length_quantiles] = 0

        calibrated_preds = p_to_calibrate - CE_estimates

        # bring the predictions back on the simplex if they move away
        return force_p_to_simplex(calibrated_preds, crop=crop)

    def plot_forest(self, include_CE_plot=True,
                    include_CE_direction_plot=True,
                    include_CE_distribution_plot=True,
                    include_data_distribution_plot=True,
                    title=None,
                    temp_param=1, d=2, folder="../figs"):

        if self.n_classes != 3:
            print("Only 3 classes plottable.")
            return

        n_plots = include_CE_plot + include_CE_direction_plot + include_CE_distribution_plot + include_data_distribution_plot
        axis_idx = 1

        if n_plots == 0:
            return

        fig = set_up_3d_fig(subplots=n_plots)
        fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0.0, right=1)

        if include_CE_plot:
            ax0, axis_idx = add_triangle_axis(fig, axis_idx, n_plots, temp_param=temp_param)
            self.__plot_CE_on_ax(ax0, fig, temp_param, d=d)

        if include_CE_direction_plot:
            ax1, axis_idx = add_triangle_axis(fig, axis_idx, n_plots, temp_param=temp_param, step=0.1)
            self.__plot_CE_direction_on_ax(ax1, fig, temp_param=temp_param)

        if include_CE_distribution_plot:
            ax2, axis_idx = add_triangle_axis(fig, axis_idx, n_plots, temp_param=temp_param)
            self.__plot_CE_distribution_on_ax(ax2, fig, temp_param=temp_param, d=d)

        if include_data_distribution_plot:
            ax3, axis_idx = add_triangle_axis(fig, axis_idx, n_plots, temp_param=temp_param)
            self.__plot_data_distribution_on_ax(ax3, fig, temp_param=temp_param)

        bs_original = np.round(np.mean(np.sum((self.original_p - self.original_y) ** 2, axis=1)), 6)
        bs_grouped = np.round(np.mean(np.sum((self.p - self.y) ** 2, axis=1)), 6)

        if hasattr(self, 'n_bins'):
            suptitle = f"CE: {np.round(self.CE_estimate(d=d), 4)}, bins: {self.n_bins}, trees: {self.n_trees}, n_data: {self.n_data}, BS: {bs_original}, BS_grouped: {bs_grouped}"
        elif hasattr(self, 'depth'):
            suptitle = f"CE: {np.round(self.CE_estimate(d=d), 4)}, depth: {self.depth}, trees: {self.n_trees}, n_data: {self.n_data}, BS: {bs_original}, BS_grouped: {bs_grouped}"

        fig.suptitle(suptitle)

        if title is not None:
            plt.savefig(folder + "/gem_" + title + ".jpg", dpi=250, bbox_inches="tight")
        else:
            plt.show()

    def __plot_CE_on_ax(self, ax, fig, temp_param, d):
        plot_ax_to_triangle_guidelines(ax)
        plot_on_triangle_guidelines(ax=ax, thick_borders=False, temp_param=temp_param)

        p = generate_points_on_triangle()
        p_scaled = un_temp_scale_points(p, temp_param)
        CE_estimates = self.CE_estimates_in_corresponding_leaves(p_scaled, d=d)

        sc = scatter_color_points_on_triangle(ax, points=p, colors=CE_estimates, use_log_colors=False)

        fig.colorbar(sc, shrink=0.5, orientation="horizontal", pad=0.01)
        ax.set_title("CE by leaves")

    def __plot_CE_direction_on_ax(self, ax, fig, temp_param):
        plot_ax_to_triangle_guidelines(ax, step=0.1)
        plot_on_triangle_guidelines(ax, thick_borders=False, temp_param=temp_param, step=0.1)

        # Arrows
        # Starts
        arrow_starts = generate_points_on_triangle(step_size=(1 - 0.02375 * 3) / 20, rounding_to=7,
                                                   start=0.02375)  # generate_points_on_triangle(step_size=0.94 / 10, rounding_to=6, start=0.02)  # generate_points_on_triangle(step_size=0.85/5, rounding_to=6, start=0.05) #step_size=0.997/10, rounding_to=6, start=0.001
        arrow_starts_scaled = un_temp_scale_points(arrow_starts, temp_param)
        # CEs
        mean_y_s = self.mean_y_in_corresponding_leaves(arrow_starts_scaled)
        mean_p_s = self.mean_p_in_corresponding_leaves(arrow_starts_scaled)
        CE_estimates = mean_p_s - mean_y_s
        # Ends
        arrow_ends_scaled = arrow_starts_scaled - CE_estimates
        arrow_ends = temp_scale_points(arrow_ends_scaled, temp_param)
        # Lengths
        arrows_scaled = arrow_ends_scaled - arrow_starts_scaled
        arrow_lengths = np.linalg.norm(arrows_scaled, axis=1)

        # Calibrated copies - errorbars
        if self.n_calibrated_copies > 0:
            # CEs
            calibrated_copies_mean_y_s = self.mean_calibrated_copies_mean_y_in_corresponding_leaves(arrow_starts_scaled)
            calibrated_copies_CE_estimates = mean_p_s - calibrated_copies_mean_y_s
            # Ends
            calibrated_copies_arrow_ends_scaled = arrow_starts_scaled - calibrated_copies_CE_estimates
            # Lengths
            calibrated_copies_arrows_scaled = calibrated_copies_arrow_ends_scaled - arrow_starts_scaled
            calibrated_copies_arrow_lengths = np.array([np.linalg.norm(calibrated_copy_arrows_scaled, axis=1)
                                                        for calibrated_copy_arrows_scaled in
                                                        calibrated_copies_arrows_scaled])

        for idx in range(len(arrow_starts)):

            if self.n_calibrated_copies > 0:

                quantile_arrow_lengths = []
                quantile_arrow_ends = []

                # Quantile arrow ends and lengths
                for quantile in [0.9, 0.95]:
                    calibrated_arrow_lengths = calibrated_copies_arrow_lengths[:, idx]
                    quantile_arrow_length = np.quantile(calibrated_arrow_lengths, quantile)

                    normalized_arrow = arrows_scaled[idx] / arrow_lengths[idx]
                    quantile_arrow_end_scaled = arrow_starts_scaled[idx] + normalized_arrow * quantile_arrow_length
                    quantile_arrow_end = temp_scale_points([quantile_arrow_end_scaled], temp_param)[0]

                    quantile_arrow_lengths.append(quantile_arrow_length)
                    quantile_arrow_ends.append(quantile_arrow_end)

                    # for i in range(10):
                    #    quantile_arrow_lengths.append(0)
                    #    quantile_arrow_end_scaled = arrow_starts_scaled[idx] + normalized_arrow * arrow_lengths[idx] * i / 9
                    #    quantile_arrow_end = un_temp_scale_points([quantile_arrow_end_scaled], temp_param)[0]
                    #    quantile_arrow_ends.append(quantile_arrow_end)

                # Drawing alpha
                if (arrow_lengths[idx] <= np.array(quantile_arrow_lengths)).all():
                    alpha = 0.05
                else:
                    alpha = 1

                # Regular arrow
                draw_classic_arrow(arrow_starts[idx], arrow_ends[idx], ax=ax, alpha=alpha)

                # Quantile arrows
                for q_idx in range(len(quantile_arrow_ends)):
                    quantile_arrow_end = quantile_arrow_ends[q_idx]
                    arw = Arrow3D([arrow_starts[idx, 0], quantile_arrow_end[0]],
                                  [arrow_starts[idx, 1], quantile_arrow_end[1]],
                                  [arrow_starts[idx, 2], quantile_arrow_end[2]],
                                  arrowstyle=ArrowStyle("|-|", widthA=0, angleA=None, widthB=0.2, angleB=None),
                                  color="red", lw=1, mutation_scale=12, zorder=19, shrinkA=0, shrinkB=0, alpha=alpha)
                    ax.add_artist(arw)

            else:
                draw_classic_arrow(arrow_starts[idx], arrow_ends[idx], ax=ax, alpha=1)

        # Background colors
        arrow_starts = generate_points_on_triangle()
        arrow_starts_scaled = un_temp_scale_points(arrow_starts, temp_param)
        mean_y_s = self.mean_y_in_corresponding_leaves(arrow_starts_scaled)
        mean_p_s = self.mean_p_in_corresponding_leaves(arrow_starts_scaled)
        CE_estimates = np.clip(((mean_p_s - mean_y_s) * 4 + 0.5), a_min=0,
                               a_max=1)  # np.clip([1,1,1] - (mean_p_s - mean_y_s) * 10, a_min=0, a_max=1)

        sc = scatter_color_points_on_triangle(ax, points=arrow_starts, colors=CE_estimates, use_log_colors=False)

        cbar = fig.colorbar(sc, shrink=0.5, orientation="horizontal", pad=0.01)
        cbar.remove()
        ax.set_title("CE direction")

    def __plot_CE_distribution_on_ax(self, ax, fig, temp_param, d):

        plot_ax_to_triangle_guidelines(ax, step=0.1)
        plot_on_triangle_guidelines(ax, thick_borders=False, temp_param=temp_param, step=0.1)

        p = generate_points_on_triangle()
        data_p = self.p
        CE_estimates_in_leaves = self.CE_estimates_in_corresponding_leaves(data_p, d=d)
        data_p_scaled = temp_scale_points(data_p, temp_param)

        # Bin data into equal width bins and sum up the CEs in each bin
        ew_tree = EqualWidthTree(20)
        binned_p = ew_tree.bin_data(p)
        binned_data_p = ew_tree.bin_data(data_p_scaled, some_other_data=CE_estimates_in_leaves)

        to_scatter_p = []
        to_scatter_ce = []
        for b0_idx, b0 in enumerate(binned_p):
            for b1_ixd, b1 in enumerate(b0):
                for b2_idx, b2 in enumerate(b1):
                    if len(b2) == 0:
                        continue
                    b_data_p = np.array(binned_data_p[b0_idx][b1_ixd][b2_idx])
                    if len(b_data_p) == 0:
                        CE_sum = 0
                    else:
                        CE_sum = np.sum(b_data_p)

                    to_scatter_ce.extend([CE_sum / self.n_data] * len(b2))
                    to_scatter_p.extend(b2)

        sc = scatter_color_points_on_triangle(ax, points=to_scatter_p, colors=to_scatter_ce / self.CE_estimate(d=d),
                                              use_log_colors=True)

        cbar = fig.colorbar(sc, shrink=0.5, orientation="horizontal", pad=0.01, format='%.0e')
        cbar.ax.tick_params(labelsize=9)

        ax.set_title("CE distribution on test data")

    def __plot_data_distribution_on_ax(self, ax, fig, temp_param):
        plot_ax_to_triangle_guidelines(ax, step=0.1)
        plot_on_triangle_guidelines(ax, thick_borders=False, temp_param=temp_param, step=0.1)

        p = generate_points_on_triangle()
        data_p = self.p
        data_p_scaled = temp_scale_points(data_p, temp_param)

        # vector_towards_viewer = (1, 1, 2**0.5 * np.tan(30 / 180 * np.pi))
        # vector_towards_viewer = vector_towards_viewer / np.linalg.norm(vector_towards_viewer)
        # data_p_towards_viewer = data_p_scaled + 0.01 * vector_towards_viewer
        # ax.scatter(data_p_towards_viewer[:,0], data_p_towards_viewer[:,1], data_p_towards_viewer[:,2], alpha=0.05, c="red", s=5)

        # Bin data into equal width bins
        ew_tree = EqualWidthTree(20)
        binned_p = ew_tree.bin_data(p)
        binned_data_p = ew_tree.bin_data(data_p_scaled)

        to_scatter_p = []
        to_scatter_n_data = []
        for b0_idx, b0 in enumerate(binned_p):
            for b1_ixd, b1 in enumerate(b0):
                for b2_idx, b2 in enumerate(b1):
                    if len(b2) == 0:
                        continue
                    b_data_p = np.array(binned_data_p[b0_idx][b1_ixd][b2_idx])
                    if len(b_data_p) == 0:
                        n_data = 0
                    else:
                        n_data = len(b_data_p)

                    to_scatter_n_data.extend([n_data / self.n_data] * len(b2))
                    to_scatter_p.extend(b2)

        sc = scatter_color_points_on_triangle(ax, points=to_scatter_p, colors=to_scatter_n_data, use_log_colors=True)

        cbar = fig.colorbar(sc, shrink=0.5, orientation="horizontal", pad=0.01, format='%.0e')
        cbar.ax.tick_params(labelsize=9)

        ax.set_title("Test data distribution")

class RandomSplitBoostingCalibrationForest(CalibrationForest):

    def __init__(self, n_trees, depth, n_classes, n_calibrated_copies=0):
        super().__init__(n_trees=n_trees, n_classes=n_classes, n_calibrated_copies=n_calibrated_copies)
        self.depth = depth

    def fit(self, p, y, random_seed=None, class_groups=None):

        super().fit(p=p, y=y, random_seed=random_seed, class_groups=class_groups)

        residual = np.copy(y)
        for _ in range(self.n_trees):
            tree = RandomSplitCalibrationTree(self.depth)
            tree.fit(p=p, y=residual, calibrated_copies_y=self.calibrated_copies_y, original_p=self.original_p,
                     original_y=self.original_y)
            self.trees.append(tree)
            residual = residual - tree.calibrate_predictions(p) + p
        """
        p_updated = np.copy(p)
        for _ in range(self.n_trees):
            tree = RandomSplitCalibrationTree(self.depth)
            tree.fit(p=p_updated, y=y, calibrated_copies_y=self.calibrated_copies_y, original_p=self.original_p, original_y=self.original_y)

            adjustment = tree.mean_y_in_corresponding_leaves(p) - tree.mean_p_in_corresponding_leaves(p)
            p_updated = p_updated + adjustment
            p_updated[p_updated > 1] = 1
            p_updated[p_updated < 0] = 0

            self.trees.append(tree)
        """
        return self

    def calibrate_predictions(self, p_to_calibrate, p_to_calibrate_at=None, crop=1e-6):
        output = np.copy(p_to_calibrate)
        #output = np.zeros(p_to_calibrate.shape)
        for tree in self.trees:
            output += tree.calibrate_predictions(p_to_calibrate) - p_to_calibrate
            #adjustment = tree.mean_y_in_corresponding_leaves(p_to_calibrate) - tree.mean_p_in_corresponding_leaves(p_to_calibrate)
            #output += adjustment
            #output[output > 1] = 1
            #output[output < 0] = 0
        return output

class RandomSplitCalibrationForest(CalibrationForest):

    def __init__(self, n_trees, depth, n_classes, n_calibrated_copies=0):
        super().__init__(n_trees=n_trees, n_classes=n_classes, n_calibrated_copies=n_calibrated_copies)
        self.depth = depth

    def fit(self, p, y, random_seed=None, class_groups=None):

        super().fit(p=p, y=y, random_seed=random_seed, class_groups=class_groups)

        for _ in range(self.n_trees):
            tree = RandomSplitCalibrationTree(self.depth)
            tree.fit(p=self.p, y=self.y, calibrated_copies_y=self.calibrated_copies_y, original_p=self.original_p,
                     original_y=self.original_y)
            self.trees.append(tree)

        return self


class EqualSizeSplitBoostingCalibrationForest(CalibrationForest):
    def __init__(self, n_trees, n_bins, n_classes, n_cv_folds=0, n_calibrated_copies=0):

        super().__init__(n_trees=n_trees, n_classes=n_classes, n_calibrated_copies=n_calibrated_copies)
        self.n_bins = n_bins
        self.n_cv_folds = n_cv_folds
        self.cv_bins = [16, 32, 64, 128, 256, 512, 1024]

    def fit(self, p, y, random_seed=None, class_groups=None):

        super().fit(p=p, y=y, random_seed=random_seed, class_groups=class_groups)

        if self.n_cv_folds > 1:
            bin_scores = find_nr_bins_with_cv(p=p, y=y, bins_to_try=self.cv_bins, n_cv_folds=self.n_cv_folds,
                                              n_trees=self.n_trees,
                                              equal_size_forest=EqualSizeSplitBoostingCalibrationForest)
            self.n_bins = self.cv_bins[np.argmin(bin_scores)]

        residual = np.copy(y)
        for _ in range(self.n_trees):
            tree = EqualSizeSplitCalibrationTree(self.n_bins)
            tree.fit(p=p, y=residual, calibrated_copies_y=self.calibrated_copies_y, original_p=self.original_p,
                     original_y=self.original_y)
            self.trees.append(tree)
            residual = residual - tree.calibrate_predictions(p)

        return self


class EqualSizeSplitCalibrationForest(CalibrationForest):
    def __init__(self, n_trees, n_bins, n_classes, n_cv_folds=0, n_calibrated_copies=0):

        super().__init__(n_trees=n_trees, n_classes=n_classes, n_calibrated_copies=n_calibrated_copies)
        self.n_bins = n_bins
        self.n_cv_folds = n_cv_folds
        self.cv_bins = [16, 32, 64, 128, 256, 512, 1024]

    def fit(self, p, y, random_seed=None, class_groups=None):

        super().fit(p=p, y=y, random_seed=random_seed, class_groups=class_groups)

        if self.n_cv_folds > 1:
            bin_scores = find_nr_bins_with_cv(p=p, y=y, bins_to_try=self.cv_bins, n_cv_folds=self.n_cv_folds,
                                              n_trees=self.n_trees, equal_size_forest=EqualSizeSplitCalibrationForest)
            self.n_bins = self.cv_bins[np.argmin(bin_scores)]

        for _ in range(self.n_trees):
            tree = EqualSizeSplitCalibrationTree(self.n_bins)
            tree.fit(p=self.p, y=self.y, calibrated_copies_y=self.calibrated_copies_y, original_p=self.original_p,
                     original_y=self.original_y)
            self.trees.append(tree)

        return self


class CalibrationTree:

    def fit(self, p, y, random_seed=None, calibrated_copies_y=None, original_p=None, original_y=None):
        assert p.shape == y.shape, "Shape of p and y doesn't match!"
        assert len(p) >= 1, "Less than 1 data point given!"

        if random_seed != None:
            np.random.seed(random_seed)

        self.n_data = len(p)
        self.n_classes = len(p[0])

    def CE_estimate(self, d=2):
        return self.root.CE_estimate(d=d) / self.n_data

    def CE_estimates_in_corresponding_leaves(self, p, d=2):
        return self.root.CE_estimates_in_corresponding_leaves(p, d=d)

    def mean_y_in_corresponding_leaves(self, p, use_original_y=False):
        return self.root.mean_y_in_corresponding_leaves(p, use_original_y=use_original_y)

    def n_data_in_corresponding_leaves(self, p):
        return self.root.n_data_in_corresponding_leaves(p)

    def mean_calibrated_copies_mean_y_in_corresponding_leaves(self, p):
        return self.root.mean_calibrated_copies_mean_y_in_corresponding_leaves(p)

    def mean_p_in_corresponding_leaves(self, p, use_original_p=False):
        return self.root.mean_p_in_corresponding_leaves(p, use_original_p=use_original_p)

    def calibrate_predictions(self, p_to_calibrate, p_to_calibrate_at=None):

        if p_to_calibrate_at is None:
            mean_y_s = self.mean_y_in_corresponding_leaves(p_to_calibrate, use_original_y=False)
            mean_p_s = self.mean_p_in_corresponding_leaves(p_to_calibrate, use_original_p=False)
        else:
            mean_y_s = self.mean_y_in_corresponding_leaves(p_to_calibrate_at, use_original_y=True)
            mean_p_s = self.mean_p_in_corresponding_leaves(p_to_calibrate_at, use_original_p=True)
        CE_estimates = mean_p_s - mean_y_s
        return p_to_calibrate - CE_estimates

    def print_tree(self):
        self.root.print_node()

    def plot_tree(self, p=None):
        if self.n_classes != 3:
            print("Only 3 classes plottable.")
            return

        fig = set_up_3d_fig()
        ax, _ = add_triangle_axis(fig, 1, 1)

        plot_ax_to_triangle_guidelines(ax)
        plot_on_triangle_guidelines(ax, thick_borders=True)

        self.root.plot_node(ax, lw=4,
                            p0_min=0, p0_max=1,
                            p1_min=0, p1_max=1,
                            p2_min=0, p2_max=1,
                            p=p)

        plt.show()


class EqualSizeSplitCalibrationTree(CalibrationTree):

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, p, y, random_seed=None, calibrated_copies_y=None, original_p=None, original_y=None):
        super().fit(p=p, y=y, random_seed=random_seed, calibrated_copies_y=calibrated_copies_y, original_p=original_p,
                    original_y=original_y)

        assert len(p) >= self.n_bins, "More data than bins!"
        bin_size_avg = len(p) / self.n_bins
        self.root = EqualSizeSplitCalibrationNode(p, y, bin_size_avg, calibrated_copies_y=calibrated_copies_y,
                                                  original_p=original_p, original_y=original_y)


class RandomSplitCalibrationTree(CalibrationTree):

    def __init__(self, depth):
        self.depth = depth
        return

    def fit(self, p, y, random_seed=None, calibrated_copies_y=None, original_p=None, original_y=None):
        super().fit(p=p, y=y, random_seed=random_seed, calibrated_copies_y=calibrated_copies_y, original_p=original_p,
                    original_y=original_y)

        self.root = RandomSplitCalibrationNode(p, y, depth_to_go=self.depth, calibrated_copies_y=calibrated_copies_y, original_p=original_p, original_y=original_y)


class EqualWidthTree:

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def bin_data(self, p, some_other_data=None):
        """
        Bin into equal width bins accroding to p.
        If some_other_data is given, then bin some_other_data. Otherwise bin p
        """

        assert len(p[0]) == 3, "Only 3 classes allowed!"

        bins = [[[[] for _ in range(self.n_bins)] for _ in range(self.n_bins)] for _ in range(self.n_bins)]
        bin_width = 1.0 / self.n_bins

        if some_other_data is None:
            some_other_data = p

        for pred_idx, pred in enumerate(p):
            idx0 = int(pred[0] // bin_width)
            idx1 = int(pred[1] // bin_width)
            idx2 = int(pred[2] // bin_width)

            if np.round(pred[0], 6) == 1.0:
                idx0 = self.n_bins - 1
            if np.round(pred[1], 6) == 1.0:
                idx1 = self.n_bins - 1
            if np.round(pred[2], 6) == 1.0:
                idx2 = self.n_bins - 1

            bins[idx0][idx1][idx2].append(some_other_data[pred_idx])

        for i in range(len(bins)):
            bins[i] = bins[i]

        return bins


class CalibrationNode:

    def __init__(self, p, calibrated_copies_y):
        self.n_classes = len(p[0])

        if calibrated_copies_y is None:
            self.n_calibrated_copies_y = 0
        else:
            self.n_calibrated_copies_y = len(calibrated_copies_y)

    def CE_estimates_in_corresponding_leaves(self, p, d=2):

        if self.split_rule is None:
            return CE_estimation_distance(self.mean_p, self.mean_y, d=d)

        output = np.zeros(len(p))

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        output[left_child_idx] = self.left_child.CE_estimates_in_corresponding_leaves(p[left_child_idx], d=d)
        output[right_child_idx] = self.right_child.CE_estimates_in_corresponding_leaves(p[right_child_idx], d=d)

        return output

    def n_data_in_corresponding_leaves(self, p):
        if self.split_rule is None:
            return self.n_data

        output = np.zeros(len(p))

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        output[left_child_idx] = self.left_child.n_data_in_corresponding_leaves(p[left_child_idx])
        output[right_child_idx] = self.right_child.n_data_in_corresponding_leaves(p[right_child_idx])

        return output

    def mean_y_in_corresponding_leaves(self, p, use_original_y=False):
        if self.split_rule is None:
            if use_original_y:
                return self.mean_original_y
            return self.mean_y

        if use_original_y:
            output = np.zeros((len(p), self.n_original_classes))
        else:
            output = np.zeros((len(p), self.n_classes))

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        output[left_child_idx] = self.left_child.mean_y_in_corresponding_leaves(p[left_child_idx],
                                                                                use_original_y=use_original_y)
        output[right_child_idx] = self.right_child.mean_y_in_corresponding_leaves(p[right_child_idx],
                                                                                  use_original_y=use_original_y)

        return output

    def mean_calibrated_copies_mean_y_in_corresponding_leaves(self, p):

        def construct_output(self, child_idx, child):
            child_output = child.mean_calibrated_copies_mean_y_in_corresponding_leaves(p[child_idx])
            if len(child_output.shape) < 3:  # Hasn't been reshaped yet
                child_n_trues = np.sum(child_idx)
                child_repeated_result = np.repeat(child_output, child_n_trues, axis=0)
                child_output = child_repeated_result.reshape(
                    (self.n_calibrated_copies_y, child_n_trues, self.n_classes))
            return child_output

        if self.split_rule is None:
            return self.calibrated_copies_mean_y

        output = np.zeros((self.n_calibrated_copies_y, len(p), self.n_classes))

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        output[:, left_child_idx] = construct_output(self=self, child_idx=left_child_idx, child=self.left_child)
        output[:, right_child_idx] = construct_output(self=self, child_idx=right_child_idx, child=self.right_child)

        return output

    def mean_p_in_corresponding_leaves(self, p, use_original_p=False):
        if self.split_rule is None:
            if use_original_p:
                return self.mean_original_p
            return self.mean_p

        if use_original_p:
            output = np.zeros((len(p), self.n_original_classes))
        else:
            output = np.zeros((len(p), self.n_classes))

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        output[left_child_idx] = self.left_child.mean_p_in_corresponding_leaves(p[left_child_idx],
                                                                                use_original_p=use_original_p)
        output[right_child_idx] = self.right_child.mean_p_in_corresponding_leaves(p[right_child_idx],
                                                                                  use_original_p=use_original_p)

        return output

    def CE_estimate(self, d=2):

        if self.split_rule is not None:
            return self.left_child.CE_estimate(d=d) + self.right_child.CE_estimate(d=d)

        return CE_estimation_distance(self.mean_p, self.mean_y, d=d) * self.n_data

    def print_node(self):

        print(self.to_string())

        if self.split_rule is not None:
            self.left_child.print_node()
            self.right_child.print_node()

    def to_string(self):

        if self.split_rule is not None:
            return f"p{self.split_on_class} >= {self.split_at}"

        return self.n_data

    def plot_node(self, ax, lw, p0_min, p0_max, p1_min, p1_max, p2_min, p2_max, p=None, zorder=1, c="red", alpha=1, lw_multiplier=0.85):

        lw = lw * lw_multiplier

        if self.split_on_class is None:
            if p is not None:
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], alpha=1, s=20)
            return

        if self.split_on_class == 0:

            ax.plot([self.split_at, self.split_at],
                    [max(1 - self.split_at - p2_max, p1_min), min(1 - self.split_at - p2_min, p1_max)],
                    [min(1 - self.split_at - p1_min, p2_max), max(1 - self.split_at - p1_max, p2_min)],
                    lw=lw, c=c, zorder=zorder, alpha=alpha)

            self.left_child.plot_node(ax, lw, p0_min, self.split_at, p1_min, p1_max, p2_min, p2_max, p=p, zorder=zorder, c=c, alpha=alpha)
            self.right_child.plot_node(ax, lw, self.split_at, p0_max, p1_min, p1_max, p2_min, p2_max, p=p, zorder=zorder, c=c, alpha=alpha)

        elif self.split_on_class == 1:

            ax.plot([max(1 - self.split_at - p2_max, p0_min), min(1 - self.split_at - p2_min, p0_max)],
                    [self.split_at, self.split_at],
                    [min(1 - self.split_at - p0_min, p2_max), max(1 - self.split_at - p0_max, p2_min)],
                    lw=lw, c=c, zorder=zorder, alpha=alpha)

            self.left_child.plot_node(ax, lw, p0_min, p0_max, p1_min, self.split_at, p2_min, p2_max, p=p, zorder=zorder, c=c, alpha=alpha)
            self.right_child.plot_node(ax, lw, p0_min, p0_max, self.split_at, p1_max, p2_min, p2_max, p=p, zorder=zorder, c=c, alpha=alpha)

        elif self.split_on_class == 2:

            ax.plot([max(1 - self.split_at - p1_max, p0_min), min(1 - self.split_at - p1_min, p0_max)],
                    [min(1 - self.split_at - p0_min, p1_max), max(1 - self.split_at - p0_max, p1_min)],
                    [self.split_at, self.split_at],
                    lw=lw, c=c, zorder=zorder, alpha=alpha)

            self.left_child.plot_node(ax, lw, p0_min, p0_max, p1_min, p1_max, p2_min, self.split_at, p=p, zorder=zorder, c=c, alpha=alpha)
            self.right_child.plot_node(ax, lw, p0_min, p0_max, p1_min, p1_max, self.split_at, p2_max, p=p, zorder=zorder, c=c, alpha=alpha)

    def _become_root(self, p, y, original_p, original_y, calibrated_copies_y):
        if original_p is None:
            original_p = np.zeros(p.shape)

        if original_y is None:
            original_y = np.zeros(y.shape)

        if calibrated_copies_y is None:
            calibrated_copies_y = np.zeros(y.shape)[np.newaxis, :]  # Filler to avoid errors without calibrated copies

        self.split_on_class = None
        self.split_at = None
        self.split_rule = None
        self.left_child = None
        self.right_child = None
        self.mean_p = np.mean(p, axis=0)
        self.mean_y = np.mean(y, axis=0)
        self.mean_original_p = np.mean(original_p, axis=0)
        self.mean_original_y = np.mean(original_y, axis=0)
        self.n_data = len(p)
        self.calibrated_copies_mean_y = np.mean(calibrated_copies_y, axis=1)
        self.n_original_classes = len(original_p[0])

    def _split_node(self, p, y, original_p, original_y, calibrated_copies_y, child_type, **kwargs):
        if original_p is None:
            original_p = np.zeros(p.shape)

        if original_y is None:
            original_y = np.zeros(y.shape)

        if calibrated_copies_y is None:
            calibrated_copies_y = np.zeros(y.shape)[np.newaxis, :]  # Filler to avoid errors without calibrated copies

        left_child_idx = ~self.split_rule(p)
        right_child_idx = self.split_rule(p)

        self.left_child = child_type(p=p[left_child_idx], y=y[left_child_idx],
                                     calibrated_copies_y=calibrated_copies_y[:, left_child_idx],
                                     original_p=original_p[left_child_idx], original_y=original_y[left_child_idx],
                                     **kwargs)
        self.right_child = child_type(p=p[right_child_idx], y=y[right_child_idx],
                                     calibrated_copies_y=calibrated_copies_y[:, right_child_idx],
                                     original_p=original_p[right_child_idx], original_y=original_y[right_child_idx],
                                     **kwargs)

        self.mean_p = None
        self.mean_y = None
        self.mean_original_p = None
        self.mean_original_y = None
        self.n_data = None
        self.calibrated_copies_mean_y = None
        self.n_original_classes = len(original_p[0])


class EqualSizeSplitCalibrationNode(CalibrationNode):

    def __init__(self, p, y, bin_size_avg, original_p=None, original_y=None, calibrated_copies_y=None):
        """
        :param p:
        :param y:
        :param original_p: not grouped p. Pass as list of np.zeros(shape.p) or None if non-existent
        :param original_y: not grouped y. Pass as list of np.zeros(shape.p) or None if non-existent
        :param calibrated_copies_y:
        """
        super().__init__(p=p, calibrated_copies_y=calibrated_copies_y)

        n_data = len(p)

        # No need to split anymore
        if n_data < int(bin_size_avg * 2):
            super()._become_root(p, y, original_p, original_y, calibrated_copies_y)
            return

        # How large chunks to split into - chosen randomly
        split_points = np.rint(np.arange(bin_size_avg, n_data, bin_size_avg))
        split_points = split_points[split_points < n_data]  # rounding error
        split_point = int(np.random.choice(split_points))

        # What class to split on - chosen randomly
        self.split_on_class = np.random.choice(np.arange(self.n_classes))
        indexes_to_sort_by = np.argsort(p[:, self.split_on_class])
        self.split_at = (p[indexes_to_sort_by[split_point - 1], self.split_on_class] + p[indexes_to_sort_by[split_point], self.split_on_class]) / 2
        self.split_rule = lambda x: x[:, self.split_on_class] >= self.split_at

        self._split_node(p=p, y=y, original_p=original_p, original_y=original_y,
                         calibrated_copies_y=calibrated_copies_y, child_type=EqualSizeSplitCalibrationNode,
                         bin_size_avg=bin_size_avg)


class RandomSplitCalibrationNode(CalibrationNode):

    def __init__(self, p, y, depth_to_go, original_p=None, original_y=None, calibrated_copies_y=None):
        """
        :param p:
        :param y:
        :param original_p: not grouped p. Pass as list of np.zeros(shape.p) or None if non-existent
        :param original_y: not grouped y. Pass as list of np.zeros(shape.p) or None if non-existent
        :param calibrated_copies_y:
        """
        super().__init__(p=p, calibrated_copies_y=calibrated_copies_y)

        # No need to split anymore
        if depth_to_go <= 0 or len(p) == 1:
            super()._become_root(p, y, original_p, original_y, calibrated_copies_y)
            return

        # What class to split on - chosen randomly
        self.split_on_class = np.random.choice(np.arange(self.n_classes))
        # Where to split - chosen randomly
        self.split_at = np.random.uniform(low=p[:, self.split_on_class].min(), high=p[:, self.split_on_class].max(), size=1)[0]
        self.split_rule = lambda x: x[:, self.split_on_class] >= self.split_at

        # Failed split - all points in one child
        if (~self.split_rule(p) == True).all() or (self.split_rule(p) == True).all():
            super()._become_root(p, y, original_p, original_y, calibrated_copies_y)
            return

        self._split_node(p=p, y=y, original_p=original_p, original_y=original_y,
                         calibrated_copies_y=calibrated_copies_y, child_type=RandomSplitCalibrationNode,
                         depth_to_go=depth_to_go-1)
