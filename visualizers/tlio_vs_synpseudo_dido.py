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

    pos_3d_plotter.add_scatter(gt_r, "ground_truth_pos", color="g")
    pos_3d_plotter.add_scatter(tlio_r, "tlio_estimated_pos", color="r")
    pos_3d_plotter.add_scatter(est_r, "syn_pseudo_estimated_pos", color="b")


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

    # multi-file selection for retrieving
    else:
        # some plotting object here to make corresponding graph
        ate_boxplotter = plotting_helpers.BoxPlotter("Absolute Translational Error (m)")
        rte_boxplotter = plotting_helpers.BoxPlotter("Relative Translational Error (m)")
        are_boxplotter = plotting_helpers.BoxPlotter("Absolute Rotational Error (rad)")
        rpe_trans_boxplotter = plotting_helpers.BoxPlotter(
            "Relative Pose Translational Error"
        )
        rpe_vel_boxplotter = plotting_helpers.BoxPlotter(
            "Relative Pose Velocity Error"
        )
        rpe_rot_boxplotter = plotting_helpers.BoxPlotter(
            "Relative Pose Rotational Error"
        )

        if args.tlio_data_list is not None:
            # retrieve overall ate spread for TLIO run
            tlio_ate_all = metrics.retrieve_ate_all(
                args,
                traj_dataloader=tlio_trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            tlio_rte_all = metrics.retrieve_rte_all(
                args,
                traj_dataloader=tlio_trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            tlio_are_all = metrics.retrieve_are_all(
                args,
                traj_dataloader=tlio_trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            # retrieve overall aggregate spreads for SynPseudo runs
            est_ate_all = metrics.retrieve_ate_all(
                args,
                traj_dataloader=estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            est_rte_all = metrics.retrieve_rte_all(
                args,
                traj_dataloader=estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            est_are_all = metrics.retrieve_are_all(
                args,
                traj_dataloader=estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            est_left_ate_all = metrics.retrieve_ate_all(
                args_left,
                traj_dataloader=left_estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            est_left_rte_all = metrics.retrieve_rte_all(
                args_left,
                traj_dataloader=left_estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )
            est_left_are_all = metrics.retrieve_are_all(
                args_left,
                traj_dataloader=left_estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            tlio_rpe_trans_all, tlio_rpe_vel_all, tlio_rpe_rot_all = metrics.retrieve_rpe_all(
                args,
                traj_dataloader=tlio_trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            est_right_rpe_trans_all, est_right_rpe_vel_all, est_right_rpe_rot_all = metrics.retrieve_rpe_all(
                args,
                traj_dataloader=estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            est_left_rpe_trans_all, est_left_rpe_vel_all, est_left_rpe_rot_all = metrics.retrieve_rpe_all(
                args_left,
                traj_dataloader=left_estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            ate_boxplotter.add_data_vec(tlio_ate_all, "TLIO")
            rte_boxplotter.add_data_vec(tlio_rte_all, "TLIO")
            are_boxplotter.add_data_vec(tlio_are_all, "TLIO")

            ate_boxplotter.add_data_vec(est_ate_all, "velocity_regressor")
            rte_boxplotter.add_data_vec(est_rte_all, "velocity_regressor")
            are_boxplotter.add_data_vec(est_are_all, "velocity_regressor")

            # ate_boxplotter.add_data_vec(est_left_ate_all, "unit_velocity_regressor")
            # rte_boxplotter.add_data_vec(est_left_rte_all, "unit_velocity_regressor")
            # are_boxplotter.add_data_vec(est_left_are_all, "unit_velocity_regressor")

            rpe_trans_boxplotter.add_data_vec(tlio_rpe_trans_all, "TLIO")
            rpe_trans_boxplotter.add_data_vec(
                est_right_rpe_trans_all, "velocity_regressor"
            )
            # rpe_trans_boxplotter.add_data_vec(est_left_rpe_trans_all, "unit_velocity_regressor")

            rpe_rot_boxplotter.add_data_vec(tlio_rpe_rot_all, "TLIO")
            rpe_rot_boxplotter.add_data_vec(est_right_rpe_rot_all, "velocity_regressor")
            # rpe_rot_boxplotter.add_data_vec(est_left_rpe_rot_all, "unit_velocity_regressor")

            rpe_vel_boxplotter.add_data_vec(tlio_rpe_vel_all, "TLIO")
            rpe_vel_boxplotter.add_data_vec(est_right_rpe_vel_all, "velocity_regressor")
            # rpe_vel_boxplotter.add_data_vec(est_left_rpe_vel_all, "unit_velocity_regressor")

            ate_boxplotter.plot()
            rte_boxplotter.plot()
            are_boxplotter.plot()
            rpe_trans_boxplotter.plot()
            rpe_vel_boxplotter.plot()
            rpe_rot_boxplotter.plot()

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
    parser.add_argument("--file_target", type=str, default=None)

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
