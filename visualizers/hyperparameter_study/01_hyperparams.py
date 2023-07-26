import argparse
from metrics_utilities import metrics

import copy
from logging_utils.argparse_utils import add_bool_arg

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import plotting_helpers

from data_loaders import data_loaders

import numpy as np
import pandas as pd
import torch


def retrieve_rpe_all(
    args_dido,
    args_blackbird,
    dido_gt_dataloader,
    blackbird_gt_dataloader,
    output_name_dido,
    output_name_blackbird,
):
    args_dido.filter_output_name = output_name_dido
    estimation_trajectories = data_loaders.GenericEstimationDataset31(args=args_dido)
    estimation_trajectory_data_loader = DataLoader(
        estimation_trajectories,
    )
    (
        dido_rpe_trans,
        _,
        _,
    ) = metrics.retrieve_rpe_all(
        args_dido,
        traj_dataloader=estimation_trajectory_data_loader,
        gt_dataloader=dido_gt_dataloader,
    )

    args_blackbird.filter_output_name = (
        output_name_blackbird
    )
    blackbird_estimation_trajectories = data_loaders.GenericEstimationDataset31(
        args=args_blackbird
    )
    blackbird_estimation_dataset_data_loader = DataLoader(
        blackbird_estimation_trajectories
    )
    (
        blackbird_rpe_trans,
        _,
        _,
    ) = metrics.retrieve_rpe_all(
        args_blackbird,
        traj_dataloader=blackbird_estimation_dataset_data_loader,
        gt_dataloader=blackbird_gt_dataloader,
    )

    return dido_rpe_trans, blackbird_rpe_trans


def plot_competing_outputs(args_dido, args_blackbird):
    # declare groundtruth DIDO dataset
    gt_trajectories_dido = data_loaders.DIDOTrajectoryDataset(args=args_dido)
    # only ever accepts a batch of 1
    gt_trajectory_dataloader_dido = DataLoader(
        gt_trajectories_dido,
    )

    # declare groundtruth blackbird dataset
    gt_trajectories_blackbird = data_loaders.BlackbirdDataset(args=args_blackbird)
    # only ever accepts a batch of 1
    gt_trajectory_dataloader_blackbird = DataLoader(
        gt_trajectories_blackbird,
    )

    args_dido.filter_output_name = "velocity_regressor_left_final_.5.txt.npy"
    estimation_trajectories = data_loaders.GenericEstimationDataset31(args=args_dido)
    estimation_trajectory_data_loader = DataLoader(
        estimation_trajectories,
    )
    (
        dido_rpe_point_5,
        _,
        _,
    ) = metrics.retrieve_rpe_all(
        args_dido,
        traj_dataloader=estimation_trajectory_data_loader,
        gt_dataloader=gt_trajectory_dataloader_dido,
    )

    args_blackbird.filter_output_name = (
        "velocity_regressor_left_final_.5_100cov.txt.npy"
    )
    blackbird_estimation_trajectories = data_loaders.GenericEstimationDataset31(
        args=args_blackbird
    )
    blackbird_estimation_dataset_data_loader = DataLoader(
        blackbird_estimation_trajectories
    )
    (
        blackbird_rpe_trans_all_point_5,
        _,
        _,
    ) = metrics.retrieve_rpe_all(
        args_blackbird,
        traj_dataloader=blackbird_estimation_dataset_data_loader,
        gt_dataloader=gt_trajectory_dataloader_blackbird,
    )

    dido_rpe_trans_point_5, blackbird_rpe_trans_point_5 = retrieve_rpe_all(
        args_dido,
        args_blackbird,
        gt_trajectory_dataloader_dido,
        gt_trajectory_dataloader_blackbird,
        "velocity_regressor_left_final_.5.txt.npy",
        "velocity_regressor_left_final_.5_100cov.txt.npy",
    )

    dido_rpe_trans_1_5, blackbird_rpe_trans_1_5 = retrieve_rpe_all(
        args_dido,
        args_blackbird,
        gt_trajectory_dataloader_dido,
        gt_trajectory_dataloader_blackbird,
        "velocity_regressor_left_final_1.5.txt.npy",
        "velocity_regressor_left_final_1.5_100cov.txt.npy",
    )

    dido_rpe_trans_2_5, blackbird_rpe_trans_2_5 = retrieve_rpe_all(
        args_dido,
        args_blackbird,
        gt_trajectory_dataloader_dido,
        gt_trajectory_dataloader_blackbird,
        "velocity_regressor_left_final_2.5.txt.npy",
        "velocity_regressor_left_final_2.5_100cov.txt.npy",
    )

    dido_rpe_trans_3_5, blackbird_rpe_trans_3_5, = retrieve_rpe_all(
        args_dido,
        args_blackbird,
        gt_trajectory_dataloader_dido,
        gt_trajectory_dataloader_blackbird,
        "velocity_regressor_left_final_3.5.txt.npy",
        "velocity_regressor_left_final_3.5_100cov.txt.npy",
    )

    # # plot boxplot for all metrics
    # fig, ax = plt.subplots(1, 3)
    # fig.suptitle("Relative Pose Errors on the DIDO Dataset")

    label = ["TLIO", "Proposed"]

    tlio_color = "purple"
    velReg_color = "green"
    overall_lw = 1.5
    overall_fs = 18

    sns.set_palette("colorblind")
    sns.set_style("whitegrid")
    sns.set_palette(["#d55e00", "#0173b2"])

    # move plots into dataframe

    plotting_df = generate_single_instance_df(
        0.5, blackbird_rpe_trans_point_5, dido_rpe_trans_point_5
    )
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(1.5, blackbird_rpe_trans_1_5, dido_rpe_trans_1_5)))
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(2.5, blackbird_rpe_trans_2_5, dido_rpe_trans_2_5)))
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(3.5, blackbird_rpe_trans_3_5, dido_rpe_trans_3_5)))

    fig_sns = plt.figure()

    ax = fig_sns.add_subplot(111)
    sns.violinplot(
        data=plotting_df,
        y="Translational RPE",
        x="Inertial Window Length",
        ax=ax,
        hue="is_dido",
        linewidth=3,
        cut = 0,
    )
    ax.set_ylim([0, 6])
    ax.set_ylabel("Relative Pose Translational Error (m)", fontsize=overall_fs)
    ax.set_xlabel("Inertial Window (s)", fontsize=overall_fs)
    ax.set_xticklabels(labels=[".5", "1.5", "2.5", "3.5"], fontsize=overall_fs)
    ax.legend(handles=ax.legend_.legend_handles, labels=["DIDO", "Blackbird"])
    plt.show()


def generate_single_instance_df(
    inertial_window_length: float,
    blackbird_rpe_trans: torch.Tensor,
    dido_rpe_trans: torch.Tensor,
):
    single_df = pd.DataFrame()

    single_df.insert(
        0,
        "Inertial Window Length",
        np.ones(blackbird_rpe_trans.size + dido_rpe_trans.size)
        * inertial_window_length,
    )
    single_df.insert(
        1, "Translational RPE", np.concatenate((blackbird_rpe_trans, dido_rpe_trans))
    )
    single_df.insert(
        2,
        "is_dido",
        np.concatenate(
            (np.ones(blackbird_rpe_trans.size), np.zeros(dido_rpe_trans.size))
        ),
    )

    return single_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sns.set_theme(context="paper", style="whitegrid")

    plt.rcParams["font.family"] = "serif"

    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_estimation_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
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
        "--filter_output_name",
        type=str,
        default="velocity_regressor_left_final_.5.txt.npy",
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

    args_dido = parser.parse_args()

    # ------------------ Blackbird parameters -----------------

    parser = argparse.ArgumentParser()
    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_estimation_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
    parser.add_argument(
        "--estimation_data_dir",
        type=str,
        default="blackbird/",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="blackbird/",
    )
    # TODO: for now, ability to enter both groundtruth and filtered TLIO output directories differently.
    parser.add_argument(
        "--tlio_data_list",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test_blackbird.txt",
    )
    parser.add_argument(
        "--data_list",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test_blackbird.txt",
    )
    parser.add_argument(
        "--data_list_loc",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/network/splits/test_blackbird.txt",
    )
    parser.add_argument(
        "--tlio_filter_output_name", type=str, default="tlio_jul5.txt.npy"
    )
    parser.add_argument(
        "--filter_output_name",
        type=str,
        default="velocity_regressor_left_final_1_100cov.txt.npy",
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
    parser.add_argument("--nominal_imu_frequency", type=float, default=400.0)

    args_blackbird = parser.parse_args()

    plot_competing_outputs(args_dido, args_blackbird)
