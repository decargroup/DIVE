import argparse

from filtering.filtering_utils import calculate_rotation_error
from metrics_utilities import metrics
from logging_utils.argparse_utils import add_bool_arg

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pylie.torch import SO3

from plotting_utils import plotting_helpers

from data_loaders import data_loaders

def plot_3_sigma_complete(args, states: dict, gt: dict):
    """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0].reshape(-1, 3)
    rs = states["r"][0].reshape(-1, 3)
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]

    fig_overall, ax_overall = plt.subplots(3, 3)
    fig_overall.suptitle(args.file_target)

    print(torch.linalg.norm((gt_r - rs), axis=0))

    plotting_helpers.three_element_3_sigma_plotter(
        ax_overall,
        0,
        "rotation",
        calculate_rotation_error(gt_C, torch.Tensor(Cs)),
        sigma_phi,
        ts,
    )
    plotting_helpers.three_element_3_sigma_plotter(
        ax_overall, 1, "position", (gt_r - rs), sigma_r, ts
    )
    plotting_helpers.three_element_3_sigma_plotter(
        ax_overall, 2, "velocity", (gt_v - vs), sigma_v, ts
    )

    pos_3d_plotter = plotting_helpers.ThreeDimensionalCartesianPlotter(
        "position_" + args.file_target
    )

    pos_3d_plotter.add_scatter(gt_r, "ground_truth_pos", color="g")
    pos_3d_plotter.add_scatter(rs, "estimated_pos", color="r")

def plot_tlio_output(args):
    trajectories = data_loaders.TLIOEstimationDataset(args=args)

    # only ever accepts a batch of 1
    trajectory_dataloader = DataLoader(
        trajectories,
    )

    # declare groundtruth dataset
    gt_trajectories = data_loaders.EuRoCDataset(args=args)

    # only ever accepts a batch of 1
    gt_trajectory_dataloader = DataLoader(
        gt_trajectories,
    )

    # run in one-shot mode with single target
    if args.file_target is not None:
        for traj_tuple, gt_tuple in zip(
            trajectory_dataloader, gt_trajectory_dataloader
        ):
            f = traj_tuple[0][0]
            traj_dict = traj_tuple[1]

            gt_dict = gt_tuple[1]

            plot_3_sigma_complete(
                args=args,
                states=traj_dict,
                gt=gt_dict,
            )

    # multi-file selection for retrieving
    else:
        # some plotting object here to make corresponding graph
        ate_boxplotter = plotting_helpers.BoxPlotter("Absolute Translational Error (m)")
        rte_boxplotter = plotting_helpers.BoxPlotter("Relative Translational Error (m)")
        are_boxplotter = plotting_helpers.BoxPlotter("Absolute Rotational Error (rad)")

        if args.tlio_data_list is not None:
            # retrieve overall ate spread for TLIO run
            tlio_ate_all = metrics.retrieve_ate_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            tlio_rte_all = metrics.retrieve_rte_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            tlio_are_all = metrics.retrieve_are_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            ate_boxplotter.add_data_vec(tlio_ate_all, "TLIO")
            rte_boxplotter.add_data_vec(tlio_rte_all, "TLIO")
            are_boxplotter.add_data_vec(tlio_are_all, "TLIO")

            ate_boxplotter.plot()
            rte_boxplotter.plot()
            are_boxplotter.plot()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------ file parameters -----------------
    parser.add_argument("--root_estimation_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument(
        "--estimation_data_dir",
        type=str,
        default="euroc/",
    )
    parser.add_argument("--root_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="euroc/",
    )
    parser.add_argument(
        "--tlio_data_list",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/data/trajectories/dataset/validation_list_short.txt",
    )
    parser.add_argument(
        "--data_list",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/data/trajectories/dataset/validation_list_short.txt",
    )
    parser.add_argument(
        "--tlio_filter_output_name", type=str, default="tlio_jul5.txt.npy"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--file_target", type=str, default="euroc_v1_02_vicon/")

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

    # ------------------ plotting parameters -----------------
    parser.add_argument("--start_idx", type=int, default= (50))

    args = parser.parse_args()

    plot_tlio_output(args)
