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
from modelling.quad import SyntheticVelocityUnitVectorRegressor

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


from os import path as osp


def retrieve_file_names(args, data_list):
    """assumes line-seperated entries for plotting targets"""
    data_list_loc = osp.join(args.root_dir, data_list)

    # retrieve trajectory paths from constructor file
    data_f = open(data_list_loc, "r")
    filenames = data_f.read().splitlines()

    return filenames


def initialize_syn_pseudo_with_dido(
    model: SyntheticVelocityUnitVectorRegressor, traj_dict: dict, idx: int
):
    # initialize initial states
    C_0 = traj_dict["C"][:, 0]
    r_0 = traj_dict["r"][:, 0]
    v_0 = traj_dict["v"][:, 0]

    # initialize initial covariance - for now, just make it equal to identity
    P_0 = torch.eye(15, 15)
    P_0[0:2, 0:2] *= args.sigma_theta_rp_init**2
    P_0[2, 2] *= args.sigma_theta_y_init**2
    P_0[3:6, 3:6] *= args.sigma_velocity_init**2
    P_0[6:9, 6:9] *= args.sigma_position_init**2
    P_0[9:12, 9:12] *= args.sigma_bias_gyro**2
    P_0[12:15, 12:15] *= args.sigma_bias_acc**2

    model.initialize(P_0.unsqueeze(0), C_0, v_0.unsqueeze(2), r_0.unsqueeze(2))


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

        quadrotor_model = SyntheticVelocityUnitVectorRegressor(args, traj_dict["v"][0].shape[0])

        # collate measurements
        measurements = torch.cat(
            (traj_dict["gyro"][0].unsqueeze(1), traj_dict["acc"][0].unsqueeze(1)), dim=1
        )

        # collate groundtruth values to be used as synthetic markers for pseudomeasurements
        markers = torch.cat(
            (
                SO3.Log(traj_dict["C"][0]),
                traj_dict["gt_omega_b"][0].unsqueeze(2),
                traj_dict["v"][0].unsqueeze(2),
                traj_dict["r"][0].unsqueeze(2),
            ),
            dim=2,
        )

        t_k_prev = traj_dict["ts"][0][0]

        for m_k, t_k, z_k in tqdm(
            zip(measurements, traj_dict["ts"][0], markers),
            total=measurements.shape[0],
        ):
            m_k = m_k.unsqueeze(0)
            z_k = z_k.unsqueeze(0)
            dt = t_k - t_k_prev
            if not quadrotor_model.initialized:
                if args.initialize_with_gt:
                    initialize_syn_pseudo_with_dido(quadrotor_model, traj_dict, i)
                else:
                    raise RuntimeError(
                        "Non-groundtruth initialization not currently implemented!"
                    )

            else:
                quadrotor_model.predict(m_k, dt)
                quadrotor_model.correct(z_k, dt)

                if args.target_elapsed_time is not None:
                    if (t_k - traj_dict["ts"][0][0]) > args.target_elapsed_time:
                        break

            # after each timestep, append aggregate vector and diagonal covariance to overall vector
            # based on marker, aggregate measurement
            omega = m_k[:, 0, :].unsqueeze(2)
            acc = m_k[:, 1, :].unsqueeze(2)
            false_marker = torch.zeros(1, 4, 1)
            if torch.any(quadrotor_model.marker):
                log_vec_k = torch.cat(
                    (
                        quadrotor_model.agg_x,
                        quadrotor_model.P.diagonal(dim1=2).unsqueeze(2),
                        t_k.view(1, 1, 1),
                        false_marker,
                        omega,
                        acc,
                        quadrotor_model.null_quad_meas.innov.view(1, 15, 1),
                    ),
                    dim=1,
                )
            else:
                log_vec_k = torch.cat(
                    (
                        quadrotor_model.agg_x,
                        quadrotor_model.P.diagonal(dim1=2).unsqueeze(2),
                        t_k.view(1, 1, 1),
                        false_marker,
                        omega,
                        acc,
                        quadrotor_model.z,
                        torch.zeros(1, 12, 1),
                    ),
                    dim=1,
                )
            # quadrotor_model.logging_vec = torch.cat(
            #     (quadrotor_model.logging_vec, log_vec_k), dim=0
            # )
            quadrotor_model.logging_vec[quadrotor_model.idx] = log_vec_k
            quadrotor_model.idx += 1

            t_k_prev = t_k

        # after having completed simulation, dump corresponding histories into numpy txt file
        # if anything remains in buffer, then dump it
        outfile = osp.join(f, args.filter_output_name)
        f_state = open(outfile, "w+")
        if quadrotor_model.logging_vec is not None:
            np.savetxt(
                f_state, quadrotor_model.logging_vec.squeeze(2).numpy(), delimiter=","
            )
        f_state.close()
        states = np.loadtxt(outfile, delimiter=",")
        np.save(outfile + ".npy", states)
        os.remove(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------ file parameters -----------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/",
    )
    parser.add_argument("--data_dir", type=str, default="data/trajectories/dataset")
    parser.add_argument(
        "--file_target",
        type=str,
        default="v_1.1_a_1.5_s_1_yaw_0.05_n_2_2021-12-29-19-06-00(0)/",
    )
    parser.add_argument("--data_list", type=str, default="test.txt")
    parser.add_argument("--filter_output_name", type=str, default="syn_pseudo_right_jacob_test.txt")
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")

    # ------------------ target parameters -----------------
    parser.add_argument("--start_idx", type=int, default=50)

    # ------------------ filtering parameters -----------------
    add_bool_arg(parser, name="use_gt", default=False)
    add_bool_arg(
        parser,
        name="self_augment",
        default=True,
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
    parser.add_argument("--sigma_gyro_ct", type=float, default=0.01)
    parser.add_argument("--sigma_accel_ct", type=float, default=0.01)
    parser.add_argument("--sigma_gyro_bias_ct", type=float, default=0.0001)
    parser.add_argument("--sigma_accel_bias_ct", type=float, default=0.0001)

    parser.add_argument("--sigma_velocity_init", type=float, default=1)
    parser.add_argument(
        "--sigma_theta_rp_init", type=float, default=30 * torch.pi / 180
    )
    parser.add_argument("--sigma_theta_y_init", type=float, default = 1e-2) # 10 * torch.pi / 180)
    parser.add_argument("--sigma_position_init", type=float, default=1)
    parser.add_argument("--sigma_bias_acc", type=float, default=0.2)
    parser.add_argument("--sigma_bias_gyro", type=float, default=1.1e-4)

    # ------------------ filtering parameters -----------------
    parser.add_argument("--target_elapsed_time", type=float, default=None)
    parser.add_argument("--dt_bar", type=float, default=1 / 400)
    parser.add_argument("--perturbation", type=str, default="right")

    # ------------------ measurement parameters -----------------
    parser.add_argument("--sigma_v_u", type=float, default=3.)
    parser.add_argument("--desired_radian_error_norm", type=float, default=.27)
    add_bool_arg(
        parser,
        name="add_noise",
        default=True,
        help="whether or not the filter is being run in a z-up frame",
    )

    # ------------------ classifying parameters -----------------
    parser.add_argument("--zero_velocity_epsilon", type=float, default=0.01)
    parser.add_argument("--zero_omega_epsilon", type=float, default=0.005)
    parser.add_argument("--zero_upward_velocity_epsilon", type=float, default=0.1)  # .1

    args = parser.parse_args()

    run_filter(args)
