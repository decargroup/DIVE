import argparse
from metrics_utilities import metrics

import copy
from logging_utils.argparse_utils import add_bool_arg

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import plotting_helpers

from data_loaders import data_loaders


def plot_3_sigma_complete(args, tlio_states: dict, est_states: dict, gt: dict):
    """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    tlio_C = tlio_states["C"][0]
    tlio_v = tlio_states["v"][0]
    tlio_r = tlio_states["r"][0]

    est_C = est_states["C"][0]
    est_v = est_states["v"][0]
    est_r = est_states["r"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]

    pos_3d_plotter = plotting_helpers.ThreeDimensionalCartesianPlotter(
        "position_" + args.file_target
    )

    pos_3d_plotter.add_scatter(gt_r, "Ground Truth", color="g")
    pos_3d_plotter.add_scatter(tlio_r, "TLIO", color="r")
    pos_3d_plotter.add_scatter(est_r, "Proposed", color="b")


def plot_competing_outputs(args, args_left):
    tlio_trajectories = data_loaders.TLIOEstimationDataset(args=args)

    estimation_trajectories = data_loaders.GenericEstimationDataset31(args=args)

    estimation_trajectories_left = data_loaders.GenericEstimationDataset31(
        args=args_left
    )

    # only ever accepts a batch of 1
    tlio_trajectory_dataloader = DataLoader(
        tlio_trajectories,
    )

    estimation_trajectory_data_loader = DataLoader(
        estimation_trajectories,
    )

    left_estimation_trajectory_data_loader = DataLoader(
        estimation_trajectories_left,
    )

    # declare groundtruth dataset
    gt_trajectories = data_loaders.DIDOTrajectoryDataset(args=args)

    # only ever accepts a batch of 1
    gt_trajectory_dataloader = DataLoader(
        gt_trajectories,
    )

    # run in one-shot mode with single target
    if args.file_target is not None:
        for tlio_traj_tuple, est_traj_tuple, gt_tuple in zip(
            tlio_trajectory_dataloader,
            estimation_trajectory_data_loader,
            gt_trajectory_dataloader,
        ):
            traj_dict = tlio_traj_tuple[1]
            gt_dict = gt_tuple[1]
            est_dict = est_traj_tuple[1]

            plot_3_sigma_complete(
                args=args,
                tlio_states=traj_dict,
                est_states=est_dict,
                gt=gt_dict,
            )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sns.set_theme(context="paper", style="whitegrid")

    # ------------------ file parameters -----------------
    parser.add_argument("--root_estimation_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument(
        "--estimation_data_dir",
        type=str,
        default="data/trajectories/dataset",
    )
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
    # TODO: for now, ability to enter both groundtruth and filtered TLIO output directories differently.
    parser.add_argument(
        "--tlio_data_list",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test.txt",
    )
    parser.add_argument(
        "--data_list_loc",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test.txt",
    )
    parser.add_argument(
        "--tlio_filter_output_name", type=str, default="tlio_jul5.txt.npy"
    )
    parser.add_argument(
        "--filter_output_name", type=str, default="velocity_regressor_left_1s.txt.npy"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--file_target", type=str, default="circle_a_2.2_v_3.5_r_1.5_yaw_2022-02-21-23-18-26/")

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

    args_left = copy.deepcopy(args)

    args_left.filter_output_name = "velocity_unit_regressor_left.txt.npy"

    plot_competing_outputs(args, args_left)
