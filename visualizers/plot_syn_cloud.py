import argparse
from data import hdf5_loader

from filtering.filtering_utils import calculate_rotation_error
from metrics_utilities import metrics

import numpy as np
import os
import h5py
from os import path as osp

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pymlg.torch import SO3

from plotting_utils import plotting_helpers

from data_loaders import data_loaders

# TODO: temporary helper for troubleshooting
def plot_position(args, states : dict, gt : dict):
    fig, ax = plt.subplots(1, 1)

    ts = states["ts"][0]
    rs = states["r"][0]
    vs = states["v"][0]
    acc = states["acc"][0]
    Cs = states["C"][0]
    bg = states["bg"][0]
    ba = states["ba"][0]
    innov = states["innov"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]
    gt_omega_b = gt["gt_omega_b"][0]
    omega = states["omega"][0]

    phi_error = calculate_rotation_error(gt_C, torch.Tensor(Cs))

    # downsampling factor
    ds = 5
    gt_v = gt_v[::ds, :]
    vs = vs[::ds, :]
    ts = ts[::ds]
    bg = bg[::ds, :]
    gt_omega_b = gt_omega_b[::ds, :]
    omega = omega[::ds, :]
    phi_error = phi_error[::ds, :]
    innov = innov[::ds, :]

    fig, ax = plt.subplots(1, 1)

    ax.plot(ts, torch.norm(gt_v, dim=1), label="v_norm")
    ax.plot(ts, torch.norm(vs, dim=1), label="v_est_norm")

    fig_1, ax_1 = plt.subplots(3, 1)

    ax_1[0].plot(ts, gt_v[:, 0], label="gt_v_x")
    ax_1[1].plot(ts, gt_v[:, 1], label="gt_v_y")
    ax_1[2].plot(ts, gt_v[:, 2], label="gt_v_z")

    ax_1[0].plot(ts, vs[:, 0], label="v_x")
    ax_1[1].plot(ts, vs[:, 1], label="v_y")
    ax_1[2].plot(ts, vs[:, 2], label="v_z")

    ax.legend(loc="upper right")
    ax_1[0].legend(loc="upper right")
    ax_1[1].legend(loc="upper right")
    ax_1[2].legend(loc="upper right")

def plot_3_sigma_complete_estimation_data(
    args, states: dict, gt: dict
):
    """
    code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt
    """

    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0]
    rs = states["r"][0]
    bg = states["bg"][0]
    ba = states["ba"][0]
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    sigma_bg = states["sigma_bg"][0]
    sigma_ba = states["sigma_ba"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)

    fig_overall, ax_overall = plt.subplots(3, 3)
    fig_overall.suptitle(args.file_target)

    plotting_helpers.three_element_3_sigma_plotter(
        ax_overall,
        0,
        "rotation",
        calculate_rotation_error(gt_C, torch.Tensor(Cs)),
        sigma_phi,
        ts,
    )
    plotting_helpers.three_element_3_sigma_plotter(ax_overall, 1, "position", (gt_r - rs), sigma_r, ts)
    plotting_helpers.three_element_3_sigma_plotter(ax_overall, 2, "velocity", (gt_v - vs), sigma_v, ts)

    pos_3d_plotter = plotting_helpers.ThreeDimensionalCartesianPlotter("position_" + args.file_target)

    pos_3d_plotter.add_scatter(gt_r, "ground_truth_pos", color="g")
    pos_3d_plotter.add_scatter(rs, "estimated_pos", color="r")

    plot_position(args, states, gt)

def plot_generic_estimation_output(args):

    # declare trajectory dataset
    trajectories = data_loaders.GenericVUEstimationDataset(args=args)

    # only ever accepts a batch of 1
    trajectory_dataloader = DataLoader(
        trajectories,
    )

    # declare groundtruth dataset
    gt_trajectories = data_loaders.CloudTrajectoryDataset(args=args)

    # only ever accepts a batch of 1
    gt_trajectory_dataloader = DataLoader(
        gt_trajectories,
    )

    # run in one-shot mode with single target
    if args.file_target is not None:

        for traj_tuple, gt_tuple in zip(trajectory_dataloader, gt_trajectory_dataloader):
            f = traj_tuple[0][0]
            traj_dict = traj_tuple[1]

            gt_dict = gt_tuple[1]

            plot_3_sigma_complete_estimation_data(
                args=args,
                states=traj_dict,
                gt=gt_dict,
            )

        plt.show()

    # multi-file selection for retrieving aggregate metrics
    elif args.data_list is not None:

        # some plotting object here to make corresponding graph
        ate_boxplotter = plotting_helpers.BoxPlotter("Absolute Translational Error (m)")
        rte_boxplotter = plotting_helpers.BoxPlotter("Relative Translational Error (m)")
        are_boxplotter = plotting_helpers.BoxPlotter("Absolute Rotational Error (rad)")

        if args.data_list is not None:
            # retrieve overall ate spread for TLIO run
            ate_all = metrics.retrieve_ate_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            rte_all = metrics.retrieve_rte_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            are_all = metrics.retrieve_are_all(
                args,
                traj_dataloader=trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            ate_boxplotter.add_data_vec(ate_all, "SynPseudo")
            rte_boxplotter.add_data_vec(rte_all, "SynPseudo")
            are_boxplotter.add_data_vec(are_all, "SynPseudo")

            ate_boxplotter.plot()
            rte_boxplotter.plot()
            are_boxplotter.plot()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry",
    )
    parser.add_argument("--data_dir", type=str, default="cloud_dataset/utiasfield_trial1/")
    parser.add_argument(
        "--data_list",
        type=str,
        default=None,
    )
    parser.add_argument("--filter_output_name", type=str, default="velRegLeft_fullAug.txt.npy")
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument(
        "--file_target",
        type=str,
        default="teach/",
    )

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=50)

    args = parser.parse_args()

    plot_generic_estimation_output(args)
