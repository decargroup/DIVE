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

import pandas as pd

import numpy as np

def retrieve_rmse(dataloader):
    proposed_rmse_ov = torch.empty(0)
    dead_reckoned_rmse_ov = torch.empty(0)

    # run in one-shot mode with single target
    for idx, (vi_failure_tuple) in enumerate(dataloader):
        f = vi_failure_tuple[0][0]
        vi_failure_dict = vi_failure_tuple[1]

        # retrieve positions 
        gt_r = vi_failure_dict["gt_r"][0]
        proposed_r = vi_failure_dict["r_proposed"][0]
        dead_reckoned_r = vi_failure_dict["r_dead_reckoned"][0]

        # compute RMSE errors between ground truth and simulated algorithms and plot
        proposed_rmse = torch.mean(torch.sqrt((proposed_r - gt_r).pow(2)), dim=1)
        dead_reckoned_rmse = torch.mean(torch.sqrt((dead_reckoned_r - gt_r).pow(2)), dim=1)

        proposed_rmse_ov = torch.cat((proposed_rmse_ov, proposed_rmse), dim=0)
        dead_reckoned_rmse_ov = torch.cat((dead_reckoned_rmse_ov, dead_reckoned_rmse), dim=0)

    return proposed_rmse_ov, dead_reckoned_rmse_ov

def plot_violins(ax_violin, dead_reckoned_rmse_ov, proposed_rmse_ov, pos):
    violin_parts = ax_violin.violinplot(
        [dead_reckoned_rmse_ov],
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.75,
        positions=[pos],
    )
    violin_parts = ax_violin.violinplot(
        [proposed_rmse_ov],
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.75,
        positions=[pos],
    )
    return violin_parts

def plot_competing_outputs(args, args_left):
    # load all desired time ranges for figure
    title = "vi_failure_position_2.txt.npy"
    args.filter_output_name = title
    vi_failure_output = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output
    )
    proposed_rmse_2, dead_reckoned_rmse_2 = retrieve_rmse(vi_failure_output_dataloader)

    title = "vi_failure_position_3.txt.npy"
    args.filter_output_name = title
    vi_failure_output_2 = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output_2
    )
    proposed_rmse_3, dead_reckoned_rmse_3 = retrieve_rmse(vi_failure_output_dataloader)

    title = "vi_failure_position_4.txt.npy"
    args.filter_output_name = title
    vi_failure_output_3 = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output_3
    )
    proposed_rmse_4, dead_reckoned_rmse_4 = retrieve_rmse(vi_failure_output_dataloader)

    title = "vi_failure_position_5.txt.npy"
    args.filter_output_name = title
    vi_failure_output_4 = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output_4
    )
    proposed_rmse_5, dead_reckoned_rmse_5 = retrieve_rmse(vi_failure_output_dataloader)

    title = "vi_failure_position_6.txt.npy"
    args.filter_output_name = title
    vi_failure_output_5 = data_loaders.VisualOdometryFailureComparisonDataset(args=args)
    vi_failure_output_dataloader = DataLoader(
        vi_failure_output_5
    )
    proposed_rmse_6, dead_reckoned_rmse_6 = retrieve_rmse(vi_failure_output_dataloader)

    labels = ["Dead Reckoning", "Proposed"]
    num_labels = ["3", "4", "5", "6"]

    tlio_color = "purple"
    velReg_color = "green"
    overall_lw = 1.5
    overall_fs = 24

    colors = ["purple", "green"]

    fig_violin, ax_violin = plt.subplots(1, 1)

    violin_parts_dead = ax_violin.violinplot(
        [dead_reckoned_rmse_3, dead_reckoned_rmse_4, dead_reckoned_rmse_5, dead_reckoned_rmse_6],
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.75,
        positions=[0, 1, 2, 3],
    )
    violin_parts_proposed = ax_violin.violinplot(
        [proposed_rmse_3, proposed_rmse_4, proposed_rmse_5, proposed_rmse_6],
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.75,
        positions=[0, 1, 2, 3],
    )

    parts = ["cmeans"]

    for idx, pc in enumerate(violin_parts_dead['bodies']):
        pc.set_facecolor("purple")
        pc.set_edgecolor('black')
        pc.set_linewidth(2)

    for idx, pc in enumerate(violin_parts_proposed['bodies']):
        pc.set_facecolor("green")
        pc.set_edgecolor('black')
        pc.set_linewidth(2)

    for partname in parts:
        vp = violin_parts_dead[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    for partname in parts:
        vp = violin_parts_proposed[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    ax_violin.set_ylabel("Translational RMSE (m)", fontsize=overall_fs)
    ax_violin.set_xticks(ticks=[0, 1, 2, 3], labels=num_labels, fontsize=overall_fs)
    ax_violin.set_xlabel("Visual-odometry failure length (s)", fontsize=overall_fs)

    labels = []
    labels.append(mpatches.Patch(color="purple", label="Dead Reckoning"))
    labels.append(mpatches.Patch(color="green", label="Proposed"))

    ax_violin.legend(handles=labels, loc="upper right", fontsize=overall_fs)

    import numpy as np

    print((torch.mean(dead_reckoned_rmse_3) - torch.mean(proposed_rmse_3)) / torch.mean(dead_reckoned_rmse_3))
    print((torch.mean(dead_reckoned_rmse_4) - torch.mean(proposed_rmse_4)) / torch.mean(dead_reckoned_rmse_4))
    print((torch.mean(dead_reckoned_rmse_5) - torch.mean(proposed_rmse_5)) / torch.mean(dead_reckoned_rmse_5))
    print((torch.mean(dead_reckoned_rmse_6) - torch.mean(proposed_rmse_6)) / torch.mean(dead_reckoned_rmse_6))

    # seaborn plotting 

    plt.legend(fontsize=overall_fs)

    sns.set_palette("colorblind")
    sns.set_style('whitegrid')
    sns.set_palette(['#d55e00', '#0173b2'])

    plotting_df = generate_single_instance_df(3, dead_reckoned_rmse_3, proposed_rmse_3)
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(4, dead_reckoned_rmse_4, proposed_rmse_4)))
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(5, dead_reckoned_rmse_5, proposed_rmse_5)))
    plotting_df = pd.concat((plotting_df, generate_single_instance_df(6, dead_reckoned_rmse_6, proposed_rmse_6)))

    fig_sns = plt.figure()

    ax = fig_sns.add_subplot(111)
    sns.violinplot(data = plotting_df, y = 'Translational RMSE', x = 'Failure Length', ax = ax, hue="is_not_proposed", linewidth=3)

    ax.set_ylim([0, 3])
    ax.set_ylabel("Relative Pose Translational Error (m)", fontsize=overall_fs)
    ax.set_xlabel("VIO Failure Length (s)", fontsize=overall_fs)
    ax.set_xticklabels(labels = ["3", "4", "5", "6"], fontsize = overall_fs)
    ax.legend(handles=ax.legend_.legend_handles, labels=['Dead Reckoning', 'DIVE'])

    fig_sns.savefig("05_drift_comp_VI.pdf", format="pdf", dpi=300)

    plt.show()

def generate_single_instance_df(failure_length : int, dead_reckoned_rmse, proposed_rmse : torch.Tensor):

    single_df = pd.DataFrame()

    single_df.insert(0, "Failure Length", np.ones(dead_reckoned_rmse.shape[0] + proposed_rmse.shape[0]) * failure_length)
    single_df.insert(1, "Translational RMSE", np.concatenate((proposed_rmse.numpy(), dead_reckoned_rmse.numpy())))
    single_df.insert(2, "is_not_proposed", np.concatenate((np.ones(proposed_rmse.shape[0]), np.zeros(dead_reckoned_rmse.shape[0]))))

    return single_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sns.set_theme(context="paper", style="whitegrid")

    plt.rcParams["font.family"] = "serif"
    plt.rc('legend',fontsize=18) # using a size in points
    plt.rcParams["ytick.labelsize"] = 24
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
        "--filter_output_name", type=str, default="vi_failure_position_1.txt.npy"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--file_target", type=str, default="v_1.9_a_4_s_1_yaw_0.05_n_3_2021-12-29-22-19-25(0)/")

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
