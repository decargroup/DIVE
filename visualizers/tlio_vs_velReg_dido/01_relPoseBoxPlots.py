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
        if args.tlio_data_list is not None:

            (
                tlio_rpe_trans_all,
                tlio_rpe_vel_all,
                tlio_rpe_rot_all,
            ) = metrics.retrieve_rpe_all(
                args,
                traj_dataloader=tlio_trajectory_dataloader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            (
                est_right_rpe_trans_all,
                est_right_rpe_vel_all,
                est_right_rpe_rot_all,
            ) = metrics.retrieve_rpe_all(
                args,
                traj_dataloader=estimation_trajectory_data_loader,
                gt_dataloader=gt_trajectory_dataloader,
            )

            # # plot boxplot for all metrics
            # fig, ax = plt.subplots(1, 3)
            # fig.suptitle("Relative Pose Errors on the DIDO Dataset")

            label = ["TLIO", "Proposed"]

            tlio_color = "purple"
            velReg_color = "green"
            overall_lw = 1.5
            overall_fs = 24

            # ax[0].boxplot(
            #     [tlio_rpe_trans_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=tlio_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [0],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[0].boxplot(
            #     [est_right_rpe_trans_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=velReg_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [1],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[0].set_ylabel("Relative Pose Translational Error (m)", fontsize=overall_fs)
            # ax[0].set_xticklabels(label, fontsize=overall_fs)

            # ax[1].boxplot(
            #     [tlio_rpe_vel_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=tlio_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [0],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[1].boxplot(
            #     [est_right_rpe_vel_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=velReg_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [1],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[1].set_ylabel("Relative Pose Velocity Error (m/s)", fontsize=overall_fs)
            # ax[1].set_xticklabels(label, fontsize=overall_fs)

            # ax[2].boxplot(
            #     [tlio_rpe_rot_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=tlio_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [0],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[2].boxplot(
            #     [est_right_rpe_rot_all],
            #     patch_artist=True,
            #     boxprops=dict(facecolor=velReg_color, color="black", linewidth=overall_lw),
            #     medianprops=dict(color="black", linewidth=overall_lw),
            #     whiskerprops=dict(linewidth=overall_lw),
            #     capprops=dict(linewidth=overall_lw),
            #     positions = [1],
            #     widths = [.75],
            #     showfliers=False,
            # )
            # ax[2].set_ylabel("Relative Pose Rotational Error (rad)", fontsize=overall_fs)
            # ax[2].set_xticklabels(label, fontsize=overall_fs)

            fig_violin, ax_violin = plt.subplots(1, 3)

            fig_violin.suptitle("Relative Pose Errors on the DIDO Dataset")

            colors = [tlio_color, velReg_color]

            violin_parts_1 = ax_violin[0].violinplot(
                [tlio_rpe_trans_all, est_right_rpe_trans_all],
                showmeans=True,
                showmedians=False,
                showextrema=False,
                widths=0.75,
                positions=[0, 1],
            )

            violin_parts_2 = ax_violin[1].violinplot(
                [tlio_rpe_vel_all, est_right_rpe_vel_all],
                showmeans=True,
                showmedians=False,
                showextrema=False,
                widths=0.75,
                positions=[0, 1],
            )

            violin_parts_3 = ax_violin[2].violinplot(
                [tlio_rpe_rot_all, est_right_rpe_rot_all],
                showmeans=True,
                showmedians=False,
                showextrema=False,
                widths=0.75,
                positions=[0, 1],
            )

            for idx, pc in enumerate(violin_parts_1['bodies']):
                pc.set_facecolor(colors[idx])
                pc.set_edgecolor('black')
                pc.set_linewidth(2)

            for idx, pc in enumerate(violin_parts_2['bodies']):
                pc.set_facecolor(colors[idx])
                pc.set_edgecolor('black')
                pc.set_linewidth(2)

            for idx, pc in enumerate(violin_parts_3['bodies']):
                pc.set_facecolor(colors[idx])
                pc.set_edgecolor('black')
                pc.set_linewidth(2)

            partnames = ['cmeans']

            for partname in partnames:
                vp = violin_parts_1[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1)

            for partname in partnames:
                vp = violin_parts_2[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1)

            for partname in partnames:
                vp = violin_parts_3[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1)

            ax_violin[0].set_ylabel("Relative Pose Translational Error (m)", fontsize=overall_fs)
            ax_violin[0].set_ylim([0, 3])
            ax_violin[1].set_ylabel("Relative Pose Velocity Error (m/s)", fontsize=overall_fs)
            ax_violin[1].set_ylim([0, 1])
            ax_violin[2].set_ylabel("Relative Pose Rotational Error (rad)", fontsize=overall_fs)
            ax_violin[2].set_ylim([0, 0.02])
            ax_violin[0].set_xticks(ticks=[0, 1], labels=label, fontsize=overall_fs)
            ax_violin[1].set_xticks(ticks=[0, 1], labels=label, fontsize=overall_fs)
            ax_violin[2].set_xticks(ticks=[0, 1], labels=label, fontsize=overall_fs)

            import numpy as np

            print((np.mean(tlio_rpe_trans_all) - np.mean(est_right_rpe_trans_all)) / np.mean(tlio_rpe_trans_all))
            print((np.mean(tlio_rpe_vel_all) - np.mean(est_right_rpe_vel_all)) / np.mean(tlio_rpe_vel_all))
            print((np.mean(tlio_rpe_rot_all) - np.mean(tlio_rpe_rot_all)) / np.mean(tlio_rpe_rot_all))

            print(np.mean(est_right_rpe_trans_all))

            import pandas as pd

            sns.set_palette("colorblind")
            sns.set_style('whitegrid')
            sns.set_palette(['#d55e00', '#0173b2'])

            fig_sns = plt.figure()
            fig_sns.tight_layout()

            # move plots into dataframe

            plotting_df = pd.DataFrame()

            plotting_df.insert(0, 'Algorithm', np.concatenate((np.zeros(tlio_rpe_trans_all.size), np.ones(est_right_rpe_trans_all.size))))

            plotting_df.insert(1, 'Relative Pose Translational Error (m)', np.concatenate((tlio_rpe_trans_all, est_right_rpe_trans_all)))
            plotting_df.insert(2, 'Relative Pose Velocity Error (m/s)', np.concatenate((tlio_rpe_vel_all, est_right_rpe_vel_all)))
            plotting_df.insert(3, 'Relative Pose Rotational Error (rad)', np.concatenate((tlio_rpe_rot_all, est_right_rpe_rot_all)))

            ax = fig_sns.add_subplot(131)
            sns.violinplot(data = plotting_df, y = 'Relative Pose Translational Error (m)', x = 'Algorithm', ax = ax, linewidth=3, cut=0, gridsize=1000)

            ax.set_ylim([0, 3])
            ax.set_ylabel("Relative Pose Translational Error (m)", fontsize=overall_fs)
            ax.set_xlabel("")
            ax.set_xticklabels(labels = ["TLIO", "DIVE"], fontsize = overall_fs + 4)
            ax.tick_params(axis='both', which='major', labelsize=15)

            ax = fig_sns.add_subplot(132)
            sns.violinplot(data = plotting_df, y = 'Relative Pose Velocity Error (m/s)', x = 'Algorithm', ax = ax, linewidth=3, cut=0, gridsize=1000)

            ax.set_ylim([0, 1])
            ax.set_ylabel("Relative Pose Velocity Error (m/s)", fontsize=overall_fs)
            ax.set_xlabel("")
            ax.set_xticklabels(labels = ["TLIO", "DIVE"], fontsize = overall_fs + 4)
            ax.tick_params(axis='both', which='major', labelsize=15)

            ax = fig_sns.add_subplot(133)
            sns.violinplot(data = plotting_df, y = 'Relative Pose Rotational Error (rad)', x = 'Algorithm', ax = ax, linewidth=3, cut=0, gridsize=1000)

            ax.set_ylim([0, 0.02])
            ax.set_ylabel("Relative Pose Rotational Error (rad)", fontsize=overall_fs)
            ax.set_xlabel("")
            ax.set_xticklabels(labels = ["TLIO", "DIVE"], fontsize = overall_fs + 4)
            ax.tick_params(axis='both', which='major', labelsize=15)

            plt.subplots_adjust(wspace=.5)

            fig_sns.savefig("01_rpe_dido.pdf", format="pdf", dpi=300)
            
            plt.show()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sns.set_theme(context="paper", style="whitegrid")

    plt.rcParams["font.family"] = "serif"
    plt.rc("figure", figsize=(16, 9))

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
        "--filter_output_name", type=str, default="velocity_regressor_left_final_1.txt.npy"
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
