import argparse
from metrics_utilities import metrics

import copy
from logging_utils.argparse_utils import add_bool_arg

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import plotting_helpers

from data_loaders import data_loaders

import torch
import matplotlib.patches as mpatches

def plot_competing_outputs(args):
    # load all desired time ranges for figure

    vi_failure_output = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output
    )

    # run in one-shot mode with single target
    for idx, (vi_failure_tuple) in enumerate(vi_failure_output_dataloader):
        f = vi_failure_tuple[0][0]
        vi_failure_dict = vi_failure_tuple[1]

        # vi failure dict contains
        gt_r = vi_failure_dict["gt_r"][0]
        proposed_r = vi_failure_dict["r_proposed"][0]
        dead_reckoned_r = vi_failure_dict["r_dead_reckoned"][0]

        # visualize trajectories
        plt.rcParams.update({"axes.titlesize": 18})

        pos_3d_plotter = plotting_helpers.ThreeDimensionalCartesianPlotter(
            ""
        )

        pos_3d_plotter.add_scatter(gt_r, "Ground Truth", color="g")
        pos_3d_plotter.add_scatter(dead_reckoned_r, "Dead Reckoning (Baseline)", color="r")
        pos_3d_plotter.add_scatter(proposed_r, "DIVE (Proposed)", color="b")

        pos_3d_plotter.fig.savefig("vi_failure_3d_vis.png", format="png", dpi=600)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sns.set_theme(context="paper", style="whitegrid")

    plt.rcParams["font.family"] = "serif"
    plt.rc("figure", figsize=(16, 9))
    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/trajectories/dataset",
    )
    parser.add_argument(
        "--data_list_loc",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test.txt",
    )
    parser.add_argument(
        "--filter_output_name", type=str, default="vi_failure_position_6_vis.txt.npy"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--file_target", type=str, default="v_0.5_a_1.5_s_1_yaw_0.05_n_2_2021-12-29-17-25-41(0)/")

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=(50))

    # ------------------ filtering parameters -----------------
    add_bool_arg(parser, name="use_gt", default=False)
    add_bool_arg(
        parser,
        name="self_augment",
        default=False,
        help="decide whether to augment ground-truth IMU data or use measured gyro/acc",
    )
    add_bool_arg(
        parser,
        name="z_up_frame",
        default=True,
        help="whether or not the filter is being run in a z-up frame",
    )

    args = parser.parse_args()

    plot_competing_outputs(args)
