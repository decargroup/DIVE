import argparse

from filtering.filtering_utils import calculate_rotation_error
from metrics_utilities import metrics
from logging_utils.argparse_utils import add_bool_arg

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import plotting_helpers

from data_loaders import data_loaders

from pymlg.torch import SO3

from filtering import filtering_utils

from network.resnet1d.loss import loss_geodesic

from pymlg.numpy import SO3 as SO3_np

import scipy.constants

def plot_network_outputs(args, states : dict, gt : dict):
    
    # form binary mask from markers
    update_marker = (states["update_marker"][0] == 1).view(-1)

    Cs = states["C"][0][update_marker]
    vs = states["v"][0][update_marker]
    rs = states["r"][0][update_marker]
    ts = states["ts"][0][update_marker]

    gt_C = gt["C"][0][update_marker]
    gt_r = gt["r"][0][update_marker]
    gt_v = gt["v"][0][update_marker]

    meas = states["meas"][0][update_marker]

    # form C_ag from orientation estimate and rotate it back into the inertial frame using gt

    c21 = Cs[:, 1, 0]
    c11 = Cs[:, 0, 0]

    # compute overall rotation for imu samples
    gamma = torch.arctan2(c21, c11)
    phi_gamma = torch.cat((torch.zeros(gamma.shape[0], 2), gamma.unsqueeze(1)), dim=1).view(-1, 3, 1)

    C_ag_hat = SO3.Exp(phi_gamma)

    c21_t = gt_C[:, 1, 0]
    c11_t = gt_C[:, 0, 0]

    # compute overall rotation for imu samples
    gamma_t = torch.arctan2(c21_t, c11_t)
    phi_gamma_t = torch.cat((torch.zeros(gamma_t.shape[0], 2), gamma_t.unsqueeze(1)), dim=1).view(-1, 3, 1)

    C_ag = SO3.Exp(phi_gamma_t)

    # rotate vectors back into inertial frame using estimated orientation, and plot arrows
    meas_a = (C_ag_hat @ meas.unsqueeze(2)).view(-1, 3)

    vu_zw_g = (C_ag.transpose(1, 2) @ gt_v.unsqueeze(2)).view(-1, 3) / torch.norm(gt_v, dim=1).reshape(-1, 1)

    vu_zw_a = gt_v / torch.norm(gt_v, dim=1).reshape(-1, 1)

    fig, ax = plt.subplots(1, 1)
    point_limit_lower = 0
    point_limit_upper = -1
    ax.scatter(gt_r[point_limit_lower:point_limit_upper, 0], gt_r[point_limit_lower:point_limit_upper, 1], label="gt_r_xy")
    # ax.scatter(rs[point_limit_lower:point_limit_upper, 0], rs[point_limit_lower:point_limit_upper, 1], label="est_r_xy")
    ax.quiver(gt_r[point_limit_lower:point_limit_upper, 0], gt_r[point_limit_lower:point_limit_upper, 1], meas_a[point_limit_lower:point_limit_upper, 0], meas_a[point_limit_lower:point_limit_upper, 1], color="r")
    ax.quiver(gt_r[point_limit_lower:point_limit_upper, 0], gt_r[point_limit_lower:point_limit_upper, 1], vu_zw_a[point_limit_lower:point_limit_upper, 0], vu_zw_a[point_limit_lower:point_limit_upper, 1], color="g")

    # generate true gravity-aligned velocity norm and compute geodesic distance and plot alongside velocity norm

    vu_zw_g = (C_ag.transpose(1, 2) @ gt_v.unsqueeze(2)).view(-1, 3) / torch.norm(gt_v, dim=1).reshape(-1, 1)

    # calculate geodesic distance
    geodesic_distance = (torch.arctan2(torch.norm(torch.cross(meas, vu_zw_g), dim=1), torch.bmm(vu_zw_g.view(-1, 1, 3), meas.view(-1, 3, 1)).view(-1)))

    mse_loss = torch.mean((meas - vu_zw_g).pow(2), dim=1)
    vel_norm_masked = torch.norm(gt_v, dim=1)

    fig_1, ax_1 = plt.subplots(1, 1)
    ax_1.plot(ts, vel_norm_masked, label="vel_norm")
    ax_1.plot(ts, geodesic_distance, label="loss")

    ax.legend(loc="upper right")
    ax_1.legend(loc="upper right")

def plot_marker_values(args, states : dict, gt : dict):
    
    fig, ax = plt.subplots(1, 1)

    ts = states["ts"][0]
    pseudo_markers = states["pseudo_markers"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]

    gt_v_b = gt_C.transpose(1, 2) @ gt_v.unsqueeze(2)

    # ax.plot(ts, pseudo_markers[:, 0], ls="-", lw=3, label="zero_ang_vel_b")
    ax.plot(ts, pseudo_markers[:, 1], ls="--", lw=3,label="zero_vel_b")
    # ax.plot(ts, pseudo_markers[:, 2], ls="-.", lw=3, label="zero_vel_b_up")
    ax.plot(ts, pseudo_markers[:, 3], ls=":", lw=3, label="zero_vel_b_lat")
    # ax.scatter(ts, torch.linalg.norm((gt_v), dim=1), label="v_norm")
    # ax.scatter(ts, gt_v_b[:, 2, :], label="v_up")
    # ax.scatter(ts, gt_v_b[:, 1, :], label="v_lat")

    ax.legend(loc="upper right")

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

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]
    acc = gt["acc"][0]
    gyro = gt["gyro"][0]
    gt_omega_b = gt["gt_omega_b"][0]
    omega = states["omega"][0]
    meas = states["meas"][0]
    innov = states["innov"][0]

    phi_error = calculate_rotation_error(gt_C, torch.Tensor(Cs))
    gt_phi = SO3.Log(gt_C)
    est_phi = SO3.Log(Cs)

    rot_error = torch.empty(gt_C.shape[0], 3, 1)
    idx = 0
    for C, C_hat in zip(gt_C, torch.Tensor(Cs)):
        err_k = torch.Tensor(SO3_np.Log((C @ C_hat.transpose(0, 1)).numpy())).view(1, 3, 1)
        rot_error[idx] = err_k
        idx += 1

    v_b = vs
    gt_v_b = gt_v.view(-1, 3, 1)

    euler_rpy_gt = SO3.to_euler(gt_C)
    euler_rpy_est = SO3.to_euler(Cs)

    # downsampling factor
    ds = 5
    v_b = v_b[::ds, :, :]
    gt_v_b = gt_v_b[::ds, :, :]
    # ts = ts[::ds]
    bg = bg[::ds, :]
    gt_omega_b = gt_omega_b[::ds, :]
    omega = omega[::ds, :]
    # phi_error = phi_error[::ds, :]
    # gt_phi = gt_phi[::ds, :]
    # est_phi = est_phi[::ds, :]
    euler_rpy_gt = euler_rpy_gt[::ds, :]
    euler_rpy_est = euler_rpy_est[::ds, :]
    Cs = Cs[::ds, :, :]
    # acc = acc[::ds, :]
    # gyro = gyro[::ds, :]
    # innov = innov[::ds, :]
    # print(innov.shape)

    # ax.scatter(ts, gt_phi[:, 2], label="gt_phi_z")
    # ax.scatter(ts, gt_phi[:, 1], label="gt_phi_y")
    # ax.scatter(ts, gt_phi[:, 0], label="gt_phi_x")
    ax.scatter(ts, est_phi[:, 2], label="est_phi_z")
    ax.scatter(ts, est_phi[:, 1], label="est_phi_y")
    # ax.scatter(ts, phi_error[:, 2], label="phi_error_z")
    # ax.scatter(ts, torch.norm(acc, dim=1), label="acc_norm")
    # ax.scatter(ts, rot_error[:, 2], label="rot_error_z")
    ax.scatter(ts, est_phi[:, 0], label="est_phi_x")
    # ax.scatter(ts, torch.linalg.norm(gt_v_b, dim=1), label="gt_v_norm")

    ax.legend(loc="upper right")

def moving_average(a, n=800):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_error(
    args, gt: dict
):
    """
    code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt
    """

    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)
    ts = gt["ts"][0]

    gyro = gt["gyro"][0].unsqueeze(2)
    acc = gt["acc"][0].unsqueeze(2)

    gt_gyro = gt["gt_omega_b"][0].unsqueeze(2)
    gt_acc_a = gt["gt_acc_a"][0].unsqueeze(2)

    gt_acc_b = gt_C.transpose(1, 2) @ (gt_acc_a - torch.tensor([0, 0, -scipy.constants.g]).view(1, 3, 1))

    omega_diff = gt_gyro - gyro
    acc_diff = gt_acc_b - acc

    omega_mean = torch.mean(omega_diff, dim=0)
    acc_mean = torch.mean(acc_diff, dim=0)

    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(moving_average(acc_diff[:, 0]), label="acc_diff_x")
    ax[0, 0].axhline(acc_mean[0], color="r")
    ax[1, 0].plot(moving_average(acc_diff[:, 1]), label="acc_diff_y")
    ax[1, 0].axhline(acc_mean[1], color="r")
    ax[2, 0].plot(moving_average(acc_diff[:, 2]), label="acc_diff_z")
    ax[2, 0].axhline(acc_mean[2], color="r")

    ax[0, 1].plot(moving_average(omega_diff[:, 0]), label="omega_diff_x")
    ax[0, 1].axhline(omega_mean[0], color="r")
    ax[1, 1].plot(moving_average(omega_diff[:, 1]), label="omega_diff_y")
    ax[1, 1].axhline(omega_mean[1], color="r")
    ax[2, 1].plot(moving_average(omega_diff[:, 2]), label="omega_diff_z")
    ax[2, 1].axhline(omega_mean[2], color="r")
    
    ax[0, 0].legend(loc="upper right")
    ax[1, 0].legend(loc="upper right")
    ax[2, 0].legend(loc="upper right")
    ax[0, 1].legend(loc="upper right")
    ax[1, 1].legend(loc="upper right")
    ax[2, 1].legend(loc="upper right")

def load_trajectories(args):

    # declare groundtruth dataset
    gt_trajectories = data_loaders.DIDOTrajectoryDataset(args=args)

    # only ever accepts a batch of 1
    gt_trajectory_dataloader = DataLoader(
        gt_trajectories,
    )

    # run in one-shot mode with single target
    if args.file_target is not None:

        for gt_tuple in gt_trajectory_dataloader:
            f = gt_tuple[0][0]
            traj_dict = gt_tuple[1]

            gt_dict = gt_tuple[1]

            plot_error(
                args=args,
                gt=gt_dict,
            )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
    parser.add_argument("--data_dir", type=str, default="data/trajectories/dataset/")
    parser.add_argument(
        "--data_list",
        type=str,
        default="validation_list_short.txt",
    )
    parser.add_argument("--filter_output_name", type=str, default="velocity_regressor_left_fixed_ga.txt.npy")
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")
    parser.add_argument(
        "--file_target",
        type=str,
        default="v_1.9_a_4_s_1_yaw_0.05_n_3_2021-12-29-22-19-25(0)",
    )

    """
    good trajectories:
    v_1.9_a_4_s_1_yaw_0.05_n_3_2021-12-29-22-19-25(0)

    bad trajectories:
    v_0.5_a_1.5_s_1_yaw_0.05_n_2_2021-12-29-17-25-41(1)
    eight_a_0.7_v_1.5_rx_1.8_yaw_2022-02-22-11-35-43/ (with innov)
    """

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

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=50)

    args = parser.parse_args()

    load_trajectories(args)
