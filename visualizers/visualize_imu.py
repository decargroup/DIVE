# %%
import argparse
from logging_utils.argparse_utils import add_bool_arg
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import time

from pymlg.torch import SE23, SO3

from modelling.imu import *
from modelling.quad import VelocityUnitVectorRegressor, VelocityVectorRegressor

from filtering.filtering_utils import calculate_rotation_error

from data import hdf5_loader

import matplotlib.pyplot as plt

from data_loaders import data_loaders
import time

def augment_data(gyro : torch.Tensor, acc : torch.Tensor):
    """
    helper function for generating the gravity-aligned input history given an initial orientation, measurements, and corresponding timestamps

    Parameters
    ----------
    ts : torch.Tensor
        The input value : torch.Tensor with shape [N]
    gyro : torch.Tensor
        Raw gyroscope measurement history : torch.Tensor with shape [3, N]
    acc : torch.Tensor
        Raw accelerometer measurement history : torch.Tensor with shape [3, N]
    C_0 : torch.Tensor
        Initial C_ba rotation matrix : torch.Tensor with shape [1, 3, 3]

    Returns
    -------
    G_{k-1} : torch.Tensor
        Auxiliary propogation matrix : torch.Tensor with shape [N, 5, 5]
    """

    # generating a uniform distribution in range [r1, r2)
    # (r1 - r2) * torch.rand() + r2    

    # accel noise parameters
    sigma_acc_ct = (6e-3 - 2e-2) * torch.rand(1) + 2e-2

    # gyro noise parameters
    sigma_gyro_ct = (1e-3 - 2e-3) * torch.rand(1) + 2e-3

    # transform into normal vector set
    # to transform into covariance
    # Q_n = Q_c / dt
    # cholesky(Q_n) @ N(0, 1) = N(0, Q_n), but cholesky of a real-matrix is just the square of the matrix

    # define dt to be 1 / nominal frequency for discrete-time noise definition
    dt = 1 / 400

    # accel noise
    acc_noise = torch.sqrt(sigma_acc_ct**2 / dt) * torch.randn(3, acc.shape[1])
    gyro_noise = torch.sqrt(sigma_gyro_ct**2 / dt) * torch.randn(3, gyro.shape[1])

    # generate initial bias estimate error
    gyro_bias = (-.05 - .05) * torch.rand(3, 1) + .05
    acc_bias = (-.2 - .2) * torch.rand(3, 1) + .2

    acc = acc + acc_noise + acc_bias
    gyro = gyro + gyro_noise + gyro_bias

    return gyro, acc

def plot_gt_acc(gt : dict, args):

    gt_C = gt["C"][0]
    gt_v = gt["v"][0]
    gt_r = gt["r"][0]
    acc = gt["acc"][0]
    gyro = gt["gyro"][0]
    gt_omega_b = gt["gt_omega_b"][0]
    gt_acc_a = gt["gt_acc_a"][0]

    # transform into body-frame acceleration
    gt_acc_b = (gt_C.transpose(1, 2) @ (gt_acc_a.unsqueeze(2) - torch.Tensor([0, 0, -scipy.constants.g]).view(1, 3, 1))).squeeze(2)

    # augment data
    gyro_aug, acc_aug = augment_data(gt_omega_b.transpose(0, 1), gt_acc_b.transpose(0, 1))

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(acc_aug[0, :], label="aug_acc_x")
    ax[0].plot(acc[:, 0], label="raw_acc_x")
    ax[0].plot(gt_acc_b[:, 0], label="gt_acc_x")

    ax[1].plot(acc_aug[1, :], label="aug_acc_y")
    ax[1].plot(acc[:, 1], label="raw_acc_y")
    ax[1].plot(gt_acc_b[:, 1], label="gt_acc_y")

    ax[2].plot(acc_aug[2, :], label="aug_acc_z")
    ax[2].plot(acc[:, 2], label="raw_acc_z")
    ax[2].plot(gt_acc_b[:, 2], label="gt_acc_z")

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")

    fig_gyro, ax_gyro = plt.subplots(3, 1)

    ax_gyro[0].plot(gyro_aug[0, :], label="aug_gyro_x")
    ax_gyro[0].plot(gyro[:, 0], label="raw_gyro_x")
    ax_gyro[0].plot(gt_omega_b[:, 0], label="gt_gyro_x")

    ax_gyro[1].plot(gyro_aug[1, :], label="aug_gyro_y")
    ax_gyro[1].plot(gyro[:, 1], label="raw_gyro_y")
    ax_gyro[1].plot(gt_omega_b[:, 1], label="gt_gyro_y")

    ax_gyro[2].plot(gyro_aug[2, :], label="aug_gyro_z")
    ax_gyro[2].plot(gyro[:, 2], label="raw_gyro_z")
    ax_gyro[2].plot(gt_omega_b[:, 2], label="gt_gyro_z")

    ax_gyro[0].legend(loc="upper right")
    ax_gyro[1].legend(loc="upper right")
    ax_gyro[2].legend(loc="upper right")
    plt.show()

def visualize_imu(args):
    # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
    torch.set_default_dtype(torch.float64)

    trajectories = data_loaders.DIDOTrajectoryDataset(args=args)

    # only ever accepts a batch of 1
    trajectory_dataloader = DataLoader(
        trajectories,
    )

    for i, traj_tuple in enumerate(trajectory_dataloader, 0):
        f = traj_tuple[0][0]
        traj_dict = traj_tuple[1]

        plot_gt_acc(traj_dict, args)

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
        "--file_target",
        type=str,
        default="eight_a_1.5_v_2.5_rx_1.8_2022-02-22-11-45-31/",
    )
    parser.add_argument("--data_list", type=str, default="test.txt")
    parser.add_argument(
        "--filter_output_name", type=str, default="velocity_regressor_left_fixed_ga.txt"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/lightning_logs/3_40_fixed_ga_v_ep5/vu-regressor-best-val-loss.ckpt", #/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/lightning_logs/3_40_full_vel_5ep/vu-regressor-best-val-loss.ckpt",
    ) 

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=50)

    # ------------------ runtime parameters -----------------
    parser.add_argument("--device", type=str, default="cuda:0")

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
        name="initialize_with_gt",
        default=True,
        help="initialize state with provided groundtruth",
    )
    add_bool_arg(
        parser,
        name="z_up_frame",
        default=True,
        help="whether or not the filter is being run in a z-up frame",
    )
    add_bool_arg(
        parser,
        name="check_numeric",
        default=False,
        help="check numeric against analytical jacobian",
    )
    add_bool_arg(
        parser,
        name="reinitialize_after_inertial_window",
        default=True,
        help="reinitialize with ground truth state after inertial window",
    )
    parser.add_argument("--sigma_gyro_ct", type=float, default=1.745e-4)
    parser.add_argument("--sigma_accel_ct", type=float, default=6e-4)
    parser.add_argument("--sigma_gyro_bias_ct", type=float, default=4.848e-5)
    parser.add_argument("--sigma_accel_bias_ct", type=float, default=1.5e-4)

    parser.add_argument("--sigma_velocity_init", type=float, default=.25)
    parser.add_argument(
        "--sigma_theta_rp_init", type=float, default=1 * torch.pi / 180
    )
    parser.add_argument(
        "--sigma_theta_y_init", type=float, default=1e-2
    )  # 10 * torch.pi / 180)
    parser.add_argument("--sigma_position_init", type=float, default=.1)
    parser.add_argument("--sigma_bias_acc", type=float, default=0.2)
    parser.add_argument("--sigma_bias_gyro", type=float, default=1.1e-4)

    parser.add_argument("--cov_scaling", type=float, default=10.)

    # ------------------ filtering parameters -----------------
    parser.add_argument("--target_elapsed_time", type=float, default=None)
    parser.add_argument("--dt_bar", type=float, default=1 / 400)
    parser.add_argument("--perturbation", type=str, default="left")

    # ------------------ imu-based parameters -----------------
    parser.add_argument("--inertial_window_length", type=float, default=3., help="desired length of inertial window in seconds")
    parser.add_argument("--nominal_imu_frequency", type=float, default=400., help="Hz")
    parser.add_argument("--update_frequency", type=float, default=20., help="Hz")

    # ------------------ measurement parameters -----------------
    parser.add_argument("--sigma_v_u", type=float, default=3)

    # ------------------ classifying parameters -----------------
    parser.add_argument("--zero_velocity_epsilon", type=float, default=0.01)
    parser.add_argument("--zero_omega_epsilon", type=float, default=0.005)
    parser.add_argument("--zero_upward_velocity_epsilon", type=float, default=0.1)  # .1

    args = parser.parse_args()

    visualize_imu(args)
