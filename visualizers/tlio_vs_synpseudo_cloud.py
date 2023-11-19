import argparse

import torch
from torch.utils.data import DataLoader

from pymlg.torch import SO3

from plotting_utils import plotting_helpers
import matplotlib.pyplot as plt

from data_loaders import data_loaders

def plot_3_sigma_complete(args, states: dict, gt: dict, est_states: dict):
    """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

    ts_min = 0
    ts_max = torch.inf

    rs = states["r"][0]
    state_ts = states["ts"][0]
    state_ts_mask = torch.where(torch.logical_and(state_ts < ts_max, state_ts > ts_min))

    est_rs = est_states["r"][0]
    est_state_ts = est_states["ts"][0]
    est_state_ts_mask = torch.where(torch.logical_and(est_state_ts < ts_max, est_state_ts > ts_min))

    gt_r = gt["raw_gps"][0]
    gt_state_ts = gt["raw_gps_ts"][0]
    gt_state_ts_mask = torch.where(torch.logical_and(gt_state_ts < ts_max, gt_state_ts > ts_min))

    pos_3d_plotter = plotting_helpers.ThreeDimensionalCartesianPlotter(
        "position_" + args.file_target
    )

    pos_3d_plotter.add_scatter(gt_r[gt_state_ts_mask], "ground_truth_pos", color="g")
    pos_3d_plotter.add_scatter(rs[state_ts_mask], "tlio_estimated_pos", color="r")
    pos_3d_plotter.add_scatter(est_rs[est_state_ts_mask], "syn_pseudo_estimated_pos", color="b")

    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(state_ts[state_ts_mask], rs[state_ts_mask][:, 0], label="tlio_estimated_pos_x")
    ax[0].scatter(est_state_ts[est_state_ts_mask], est_rs[est_state_ts_mask][:, 0], label="syn_pseudo_estimated_pos_x")
    ax[0].scatter(gt_state_ts[gt_state_ts_mask], gt_r[gt_state_ts_mask][:, 0], label="ground_truth_pos_x")
    ax[0].legend(loc="upper right")

    ax[1].scatter(state_ts[state_ts_mask], rs[state_ts_mask][:, 1], label="tlio_estimated_pos_y")
    ax[1].scatter(est_state_ts[est_state_ts_mask], est_rs[est_state_ts_mask][:, 1], label="syn_pseudo_estimated_pos_y")
    ax[1].scatter(gt_state_ts[gt_state_ts_mask], gt_r[gt_state_ts_mask][:, 1], label="ground_truth_pos_y")
    ax[1].legend(loc="upper right")

    ax[2].scatter(state_ts[state_ts_mask], rs[state_ts_mask][:, 2], label="tlio_estimated_pos_z")
    ax[2].scatter(est_state_ts[est_state_ts_mask], est_rs[est_state_ts_mask][:, 2], label="syn_pseudo_estimated_pos_z")
    ax[2].scatter(gt_state_ts[gt_state_ts_mask], gt_r[gt_state_ts_mask][:, 2], label="ground_truth_pos_z")
    ax[2].legend(loc="upper right")

def plot_tlio_output(args, est_args):

    # declare trajectory dataset
    est_trajectories = data_loaders.GenericEstimationDataset31(args=est_args)
    est_trajectory_dataloader = DataLoader(
        est_trajectories,
    )

    trajectories = data_loaders.TLIOEstimationDataset(args=args)
    trajectory_dataloader = DataLoader(
        trajectories,
    )

    # declare groundtruth dataset
    gt_trajectories = data_loaders.CloudTrajectoryDataset(args=args)
    gt_trajectory_dataloader = DataLoader(
        gt_trajectories,
    )

    # run in one-shot mode with single target
    if args.file_target is not None:
        for traj_tuple, gt_tuple, est_traj_tuple in zip(
            trajectory_dataloader, gt_trajectory_dataloader, est_trajectory_dataloader
        ):
            f = traj_tuple[0][0]
            traj_dict = traj_tuple[1]
            est_traj_dict = est_traj_tuple[1]

            gt_dict = gt_tuple[1]

            plot_3_sigma_complete(
                args=args,
                states=traj_dict,
                gt=gt_dict,
                est_states=est_traj_dict
            )
    else:
        raise NotImplementedError("TODO: implement multi-target plotting mode for cloud dataset.")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    cloud_target = "utiascircle_trial1/"
    instance_target = "repeat/"

    # ------------------ file parameters -----------------
    parser.add_argument("--root_estimation_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument(
        "--estimation_data_dir",
        type=str,
        default="cloud_dataset/" + cloud_target,
    )
    parser.add_argument("--root_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cloud_dataset/" + cloud_target,
    )
    parser.add_argument(
        "--tlio_filter_output_name", type=str, default="not_vio_state.txt.npy"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--file_target", type=str, default= instance_target)

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default = 50)

    # ESTIMATION DATASET PARAMETERS

    estimation_parser = argparse.ArgumentParser()

    # ------------------ file parameters -----------------
    estimation_parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry",
    )
    estimation_parser.add_argument("--data_dir", type=str, default="cloud_dataset/" + cloud_target)
    estimation_parser.add_argument(
        "--data_list",
        type=str,
        default="data/trajectories/dataset/validation_list_short.txt",
    )
    estimation_parser.add_argument("--filter_output_name", type=str, default="syn_pseudo_left_forwards_utias_field.txt.npy")
    estimation_parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    estimation_parser.add_argument(
        "--file_target",
        type=str,
        default=instance_target,
    )

    args = parser.parse_args()
    est_args = estimation_parser.parse_args()

    plot_tlio_output(args, est_args)
