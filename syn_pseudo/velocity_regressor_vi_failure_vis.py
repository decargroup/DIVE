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

from data_loaders import data_loaders


def plot_helper(ax, a, b, timestamps, e, p, label):
    ax[a, b].scatter(timestamps, e, label=label)
    ax[a, b].fill_between(
        timestamps.reshape(
            -1,
        ),
        (3 * p).reshape(
            -1,
        ),
        (-3 * p).reshape(
            -1,
        ),
        color="g",
        alpha=0.2,
    )
    ax[a, b].legend(loc="upper right")

def augment_data(gyro: torch.Tensor, acc: torch.Tensor, args):
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

    # if self-augmenting, that means that the unbiased history has been returned and must be augmented with full noise

    # generating a uniform distribution in range [r1, r2)
    # (r1 - r2) * torch.rand() + r2

    # accel noise parameters
    sigma_acc_ct = torch.Tensor([1.5e-3])

    # gyro noise parameters
    sigma_gyro_ct = torch.Tensor([1e-4])

    # transform into normal vector set
    # to transform into covariance
    # Q_n = Q_c / dt
    # cholesky(Q_n) @ N(0, 1) = N(0, Q_n), but cholesky of a real-matrix is just the square of the matrix

    # define dt to be 1 / nominal frequency for discrete-time noise definition
    dt = 1 / args.nominal_imu_frequency

    # accel noise
    acc_noise = torch.sqrt(sigma_acc_ct**2 / dt) * torch.randn(3, acc.shape[1])
    gyro_noise = torch.sqrt(sigma_gyro_ct**2 / dt) * torch.randn(3, gyro.shape[1])

    # generate initial bias estimate error
    gyro_bias = (-0.01 - 0.01) * torch.rand(3, 1) + 0.01
    acc_bias = (-0.05 - 0.05) * torch.rand(3, 1) + 0.05

    # generate axis misalignment rotation
    phi_ax_misalignment = torch.rand(1, 3, 1)
    phi_ax_misalignment = phi_ax_misalignment / torch.norm(phi_ax_misalignment)
    phi_ax_misalignment *= 5 * torch.pi / 180
    C_ax_misalignment = SO3.Exp(phi_ax_misalignment)

    acc = acc + acc_noise + acc_bias
    gyro = gyro + gyro_noise + gyro_bias

    return gyro.transpose(0, 1), acc.transpose(0, 1)


from os import path as osp


def retrieve_file_names(args, data_list):
    """assumes line-seperated entries for plotting targets"""
    data_list_loc = osp.join(args.root_dir, data_list)

    # retrieve trajectory paths from constructor file
    data_f = open(data_list_loc, "r")
    filenames = data_f.read().splitlines()

    return filenames


def initialize_with_gt(
    model : VelocityVectorRegressor, gt_k : torch.Tensor, idx: int
):

    # pull out ground-truth values from gt_k
    print(gt_k.shape)
    gt_phi = gt_k[:, 0]
    C_0 = SO3.Exp(gt_phi)
    v_0 = gt_k[:, 1].view(1, 3)
    r_0 = gt_k[:, 2].view(1, 3)

    # initialize initial covariance - for now, just make it equal to identity
    P_0 = torch.eye(15, 15)
    P_0[0:3, 0:3] *= args.sigma_theta_rp_init**2
    P_0[3:6, 3:6] *= args.sigma_velocity_init**2
    P_0[6:9, 6:9] *= args.sigma_position_init**2
    P_0[9:12, 9:12] *= args.sigma_bias_gyro**2
    P_0[12:15, 12:15] *= args.sigma_bias_acc**2

    model.initialize(P_0.unsqueeze(0), C_0, v_0.unsqueeze(2), r_0.unsqueeze(2))

import time

def run_filter(args):
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

        # collate measurements (this yields an [N, 6] tensor)
        measurements = torch.cat(
            (traj_dict["gyro"][0], traj_dict["acc"][0]), dim=1
        )

        # collate groundtruth values to be used as synthetic markers for pseudomeasurements
        # this yields an [N, 9] tensor
        gt_collate = torch.cat(
            (
                SO3.Log(traj_dict["C"][0]).squeeze(2),
                traj_dict["v"][0],
                traj_dict["r"][0],
            ),
            dim=1,
        )

        ts_all = traj_dict["ts"][0]

        # compute window length and step size
        window_index_length = int(args.N / args.dt_bar)
        discrete_step = int(args.N_step / args.dt_bar)

        # unfold measurements and gt
        measurements = measurements.unfold(0, window_index_length, discrete_step)
        gt_collate = gt_collate.unfold(0, window_index_length, discrete_step)
        ts_all = ts_all.unfold(0, window_index_length, discrete_step)

        # gt_collate.shape : [num_segments, 9, N_discrete_length]
        # measurements.shape : [num_segments, 6, N_discrete_length]
        # ts_all.shape : [num_segments, N_discrete_length]

        # declare overall logging vector
        r_final_traj = torch.empty(0, 9)

        # collate groundtruth values to be used as synthetic markers for pseudomeasurements

        # segment id
        seg_idx = 0

        for m_seg, gt_seg, ts_seg in zip(measurements, gt_collate, ts_all):

            # m_seg : [6, N_discrete_length]
            # gt_seg : [9, N_discrete_length]

            gyro = m_seg[:3, :].transpose(0, 1)
            acc = m_seg[3:, :].transpose(0, 1)

            # augment gyroscope and accelerometer measurements with noise and bias

            # gyro : [N_discrete_length, 3]
            # acc : [N_discrete_length, 3]

            gyro, acc = augment_data(gyro.transpose(0, 1), acc.transpose(0, 1), args)

            # reshape measurement and ground truth segments into original shape and feed into estimator
            # collate measurements
            measurements = torch.cat(
                (gyro.unsqueeze(1), acc.unsqueeze(1)), dim=1
            )

            phi_gt = gt_seg[:3, :].transpose(0, 1)
            v_gt = gt_seg[3:6, :].transpose(0, 1)
            r_gt = gt_seg[6:, :].transpose(0, 1)

            # phi : [N_discrete_length, 3]
            # v : [N_discrete_length, 3]
            # r : [N_discrete_length, 3]

            # collate groundtruth values to be used as synthetic markers for pseudomeasurements
            gt_collate = torch.cat(
                (
                    phi_gt.unsqueeze(2),
                    v_gt.unsqueeze(2),
                    r_gt.unsqueeze(2),
                ),
                dim=2,
            )

            ts_seg = ts_seg.view(-1)

            t_k_prev = ts_seg[0]

            quadrotor_model_proposed = VelocityVectorRegressor(args, traj_dict["v"][0].shape[0])
            quadrotor_model_dead_reckoned = VelocityVectorRegressor(args, traj_dict["v"][0].shape[0])

            for m_k, t_k, gt_k in tqdm(
                zip(measurements, ts_seg, gt_collate),
                total=measurements.shape[0],
            ):
                m_k = m_k.unsqueeze(0)
                dt = t_k - t_k_prev
                if not quadrotor_model_proposed.initialized:
                    if args.initialize_with_gt:
                        initialize_with_gt(quadrotor_model_proposed, gt_k, i)
                        initialize_with_gt(quadrotor_model_dead_reckoned, gt_k, i)
                    else:
                        raise RuntimeError(
                            "Non-groundtruth initialization not currently implemented!"
                        )

                else:
                    # perform prediction and update for proposed model
                    quadrotor_model_proposed.predict(m_k, dt)
                    quadrotor_model_proposed.correct(m_k, t_k, gt_k, dt)

                    # just dead reckon for pure prediction model
                    quadrotor_model_dead_reckoned.predict(m_k, dt)

                    if args.target_elapsed_time is not None:
                        if (t_k - traj_dict["ts"][0][0]) > args.target_elapsed_time:
                            break

                t_k_prev = t_k

                # if timespan within desired range, add to logging vector and exit
                if (seg_idx == args.target_seg_idx):
                    # generate final states of both proposed and dead reckoned models
                    _, _, r_proposed_final = SE23.to_components(quadrotor_model_proposed.x[0])

                    _, _, r_dead_reckoned_final = SE23.to_components(quadrotor_model_dead_reckoned.x[0])

                    gt_r = gt_k[:, 2].view(1, 3)

                    # at the end of every segment, append final positions to overall logging vector
                    r_vec_seg = torch.cat((gt_r.view(1, 3), r_proposed_final.squeeze(2), r_dead_reckoned_final.squeeze(2)), dim=1)

                    r_final_traj = torch.cat((r_final_traj, r_vec_seg), dim=0)

                elif (seg_idx > args.target_seg_idx):
                    # after having completed simulation, dump corresponding histories into numpy txt file
                    # if anything remains in buffer, then dump it
                    outfile = osp.join(f, args.filter_output_name)
                    f_state = open(outfile, "w+")
                    np.savetxt(
                        f_state, r_final_traj.numpy(), delimiter=","
                    )
                    f_state.close()
                    states = np.loadtxt(outfile, delimiter=",")
                    np.save(outfile + ".npy", states)
                    os.remove(outfile)

                    # exit code afterwards
                    exit()

            seg_idx += 1


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
        default=None,
    )
    parser.add_argument("--data_list", type=str, default="test.txt")
    parser.add_argument(
        "--filter_output_name", type=str, default="vi_failure_position_6_vis.txt"
    )
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")
    parser.add_argument("--data_list_loc", type=str, default="network/splits/test.txt")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/lightning_logs/validation_checkpoints/final_velReg_augment_1_best_val_loss.ckpt",
    ) 

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=50)

    # ------------------ runtime parameters -----------------
    parser.add_argument("--device", type=str, default="cuda:0")

    # ------------------ timing parameters -----------------
    parser.add_argument("--target_seg_idx", type=int, default=1)

    # ------------------ filtering parameters -----------------
    add_bool_arg(parser, name="use_gt", default=True)
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
    add_bool_arg(
        parser,
        name="contains_bias",
        default=False,
        help="whether or not the ground-truth contains a bias optimization result",
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

    parser.add_argument("--cov_scaling", type=float, default=100.)

    # ------------------ filtering parameters -----------------
    parser.add_argument("--target_elapsed_time", type=float, default=None)
    parser.add_argument("--dt_bar", type=float, default=1 / 400)
    parser.add_argument("--perturbation", type=str, default="left")

    # ------------------ imu-based parameters -----------------
    parser.add_argument("--inertial_window_length", type=float, default=1., help="desired length of inertial window in seconds")
    parser.add_argument("--nominal_imu_frequency", type=float, default=400., help="Hz")
    parser.add_argument("--update_frequency", type=float, default=20., help="Hz")
    parser.add_argument("--N", type=float, default=7., help="secs")
    parser.add_argument("--N_step", type=float, default=7., help="Hz")

    # ------------------ measurement parameters -----------------
    parser.add_argument("--sigma_v_u", type=float, default=3)

    # ------------------ classifying parameters -----------------
    parser.add_argument("--zero_velocity_epsilon", type=float, default=0.01)
    parser.add_argument("--zero_omega_epsilon", type=float, default=0.005)
    parser.add_argument("--zero_upward_velocity_epsilon", type=float, default=0.1)  # .1

    args = parser.parse_args()

    run_filter(args)
