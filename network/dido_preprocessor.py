# %%
import torch
from network.preprocessing import data_encoder, data_segmenter
from data import imu_preprocessing, hdf5_loader
import os
import scipy.constants
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import path as osp

from pymlg.torch import SO3

from filtering import filtering_utils


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
    if args.self_augment:
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

    # if not self-augmenting, that means that the "biased" or native IMU history has been returned and should be slightly randomized
    else:
        # accel noise
        acc_noise = torch.sqrt(sigma_acc_ct**2 / dt) * torch.randn(3, acc.shape[1])
        gyro_noise = torch.sqrt(sigma_gyro_ct**2 / dt) * torch.randn(3, gyro.shape[1])

        # generate initial bias estimate error
        gyro_bias = (-0.01 - 0.01) * torch.rand(3, 1) + 0.01
        acc_bias = (-0.05 - 0.05) * torch.rand(3, 1) + 0.05

        acc = acc + acc_noise + acc_bias
        gyro = gyro + gyro_bias

    return gyro, acc


def generate_gravity_aligned_input(
    ts: torch.Tensor, gyro: torch.Tensor, acc: torch.Tensor, C_0: torch.Tensor
):
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
    # compute overall rotation for imu samples
    # anchor_euler = SO3.to_euler(C_0, order="123")
    # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))

    c21 = C_0[:, 1, 0]
    c11 = C_0[:, 0, 0]

    # # compute overall rotation for imu samples
    # anchor_euler = SO3.to_euler(C_k, order="123")
    # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
    gamma = torch.arctan2(c21, c11)
    phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

    C_gamma = SO3.Exp(phi_gamma)

    # applying error to initial orientation estimate and then backwards-integration to get estimates
    # as we would do in estimation
    delta_phi = torch.rand(1, 3, 1)
    delta_phi = delta_phi / torch.norm(delta_phi)

    gyro_flipped = torch.flip(gyro.transpose(0, 1), dims=(0,))

    ts_diff_flipped = torch.flip(torch.diff(ts), dims=(0,))

    batch_delta_c = SO3.Exp(
        gyro_flipped[1:].unsqueeze(2) @ ts_diff_flipped.view(-1, 1, 1)
    )

    C_cum = [(C_0).squeeze(0)]

    for d_C in batch_delta_c:
        C_cum.append(C_cum[-1] @ d_C.transpose(0, 1))

    C_cum = torch.flip(torch.stack(C_cum), dims=(0,))

    C_g0_b = C_gamma.transpose(1, 2) @ C_cum

    acc_ga = (C_g0_b @ acc.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)

    gyro_ga = (C_g0_b @ gyro.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)

    # remove gravity from acc_ga
    acc_ga = acc_ga - torch.Tensor([0, 0, scipy.constants.g]).view(3, 1)

    phi_g0_b = SO3.Log(C_g0_b).squeeze(2).transpose(0, 1)

    return torch.cat((phi_g0_b, acc_ga), dim=0)


def load_data_and_encoding(args, f, N, freq, dt):
    # retrieve raw trajectory data
    traj_dict = hdf5_loader.retrieve_trajectory(f, start_idx=args.start_idx)

    # convert quaternions to rotation vector
    gt_phi = SO3.Log(SO3.from_quat(traj_dict["gt_q"], ordering="wxyz")).squeeze(2)

    # if retrieving ground truth data, generate gt body frame orientation and acceleration
    if args.self_augment:
        gyro = traj_dict["gt_omega_b"]
        acc = imu_preprocessing.rotate_gt_acc(
            traj_dict["gt_acc_a"], traj_dict["gt_q"]
        ).squeeze(2)
    else:
        gyro = traj_dict["gyro"]
        acc = traj_dict["acc"]

    # aggregate complete trajectory to then segment collectively
    agg_traj = torch.cat(
        (gt_phi, traj_dict["gt_v"], gyro, acc, traj_dict["ts"].unsqueeze(1)), dim=1
    )

    # segment measurements based on specification
    data_segments = data_segmenter.segment_imu_data(
        measurements=agg_traj, N=N, freq=freq, dt=dt
    )

    # mask data segments based on moments of low-velocity
    # gt_v_segmented = data_segments[:, 3:6, :]
    # vel_initial_mag, vel_final_mag = data_encoder.retrieve_velocity_magnitude(gt=gt_v_segmented)

    # velocity_mag_mask = torch.logical_or((vel_final_mag > args.velocity_mag_threshold), (vel_initial_mag > args.velocity_mag_threshold))

    # retrieve gyro and accelerometer measurements from unfolded segments
    # measurement_segments = data_segments[:, :12, :]

    # # retrieve ground truth velocity unit vector encoding
    # # this will return a [num_segments, 3] torch.Tensor
    # gt_encoding = data_encoder.retrieve_velocity_unit_vector_encoding(gt=data_segments[:, 0:6, :])

    # # retrieve segmented timestamps
    # ts_segments = data_segments[:, 12, :]

    # return measurement segments and ground truth encoding
    return data_segments


class VelocityUnitVectorDataset(Dataset):
    def __init__(self, f, N, freq, dt, args) -> Dataset:
        # retrieve trajectory paths from constructor file
        data_f = open(f, "r")
        self.filenames = data_f.read().splitlines()

        # initialize args
        self.args = args

        # initialize segmentation values
        self.N = N
        self.freq = freq
        self.dt = dt

        self.features = torch.empty((0, 13, int(N / dt)))
        # self.encoding = torch.empty((0, 3))
        # self.ts = torch.empty((0, 1, int(N / dt)))
        for f in self.filenames:
            data_loc = osp.join(f, args.ground_truth_output_name)
            features = load_data_and_encoding(args, data_loc, N, freq, dt)
            self.features = torch.cat((self.features, features), dim=0)
            # self.encoding = torch.cat((self.encoding, encoding), dim=0)
            # self.ts = torch.cat((self.ts, ts.unsqueeze(1)), dim=0)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # here, eventually need to augment ground truth data with white noise and bias
        # imu_interpolator = interpolate.interp1d(self.ts[idx, 0, :], self.features[idx, :, :], axis=1, fill_value="extrapolate")

        # interp_ts = torch.linspace(self.ts[idx, 0, 0], self.ts[idx, 0, 0] + self.N, int(self.N / self.dt))

        if self.args.frame_target == "body_referenced":
            # retrieve specific batch features pertaining to idx
            gt_phi_0 = self.features[idx, 0:3, 0]
            gt_C_0 = SO3.Exp(gt_phi_0)
            gt_v = self.features[idx, 3:6, -1]
            gyro = self.features[idx, 6:9, :]
            acc = self.features[idx, 9:12, :]
            ts = self.features[idx, 12, :]

            # compute overall rotation for imu samples
            C_b0_b = gt_C_0.transpose(1, 2) @ SO3.Exp(
                self.features[idx, 0:3, :].transpose(0, 1)
            )

            # rotate gyro and acc into the body-referenced navigation frame
            gyro_br = (
                (C_b0_b @ gyro.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)
            )
            acc_br = (
                (C_b0_b @ acc.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)
            )

            # rotate ground truth velocity into the body-referenced navigation frame
            gt_v_br = (gt_C_0.transpose(1, 2) @ gt_v.reshape(1, 3, 1)).reshape(3)

            # normalize velocity vector and form rotation vector based on rotation between [1, 0, 0] vector in corresponding frame
            gt_v_br = gt_v_br / torch.norm(gt_v_br)

            C_vu_base = filtering_utils.unit_vec_rodrigues(
                torch.Tensor([1, 0, 0]).view(1, 3), gt_v_br.view(1, 3)
            )

            phi_base_vu = SO3.Log(C_vu_base).squeeze(0)

            phi_b0_b = SO3.Log(C_b0_b).squeeze(2).transpose(0, 1)

            return (torch.cat((phi_b0_b, acc_br), dim=0), gt_v_br.view(1, 3))

        elif self.args.frame_target == "current_k_gravity_aligned":
            with torch.no_grad():
                # retrieve specific batch features pertaining to idx
                gt_phi_0 = self.features[idx, 0:3, -1]
                gt_C_0 = SO3.Exp(gt_phi_0)
                gt_v_initial = self.features[idx, 3:6, 0]
                gt_v = self.features[idx, 3:6, -1]
                gyro = self.features[idx, 6:9, :]
                acc = self.features[idx, 9:12, :]
                ts = self.features[idx, 12, :]

                c21 = gt_C_0[:, 1, 0]
                c11 = gt_C_0[:, 0, 0]

                # # compute overall rotation for imu samples
                # anchor_euler = SO3.to_euler(C_k, order="123")
                # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
                gamma = torch.arctan2(c21, c11)
                phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

                C_gamma = SO3.Exp(phi_gamma)

                gyro, acc = augment_data(gyro=gyro, acc=acc, args=self.args)

                model_input = generate_gravity_aligned_input(ts, gyro, acc, gt_C_0)

                # rotate ground truth velocity into the local gravity-aligned frame
                gt_v_ga_raw = (C_gamma.transpose(1, 2) @ gt_v.reshape(1, 3, 1)).reshape(
                    3
                )

                # normalize velocity vector and form rotation vector based on rotation between [1, 0, 0] vector in corresponding frame
                gt_v_ga = gt_v_ga_raw / torch.norm(gt_v_ga_raw)

                if self.args.train_raw_velocity:
                    return (
                        model_input,
                        gt_v_ga_raw.view(1, 3),
                        # torch.norm(gt_v),
                        # torch.norm(gt_v_initial),
                    )
                else:
                    return (
                        model_input,
                        gt_v_ga.view(1, 3),
                        # torch.norm(gt_v),
                        # torch.norm(gt_v_initial),
                    )

        elif self.args.frame_target == "gravity_aligned":
            # retrieve specific batch features pertaining to idx
            gt_phi_0 = self.features[idx, 0:3, 0]
            gt_C_0 = SO3.Exp(gt_phi_0)
            gt_v_initial = self.features[idx, 3:6, 0]
            gt_v = self.features[idx, 3:6, -1]
            gyro = self.features[idx, 6:9, :]
            acc = self.features[idx, 9:12, :]
            ts = self.features[idx, 12, :]

            # compute overall rotation for imu samples
            anchor_euler = SO3.to_euler(gt_C_0, order="123")
            C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            C_g0_b = C_gamma.transpose(1, 2) @ SO3.Exp(
                self.features[idx, 0:3, :].transpose(0, 1)
            )

            # rotate gyro and acc into the local gravity-aligned frame
            gyro_ga = (
                (C_g0_b @ gyro.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)
            )
            acc_ga = (
                (C_g0_b @ acc.transpose(0, 1).unsqueeze(2)).squeeze(2).transpose(0, 1)
            )

            # remove gravity from acc_ga
            acc_ga = acc_ga - torch.Tensor([0, 0, scipy.constants.g]).view(3, 1)

            # rotate ground truth velocity into the local gravity-aligned frame
            gt_v_ga_raw = (C_gamma.transpose(1, 2) @ gt_v.reshape(1, 3, 1)).reshape(3)

            # normalize velocity vector and form rotation vector based on rotation between [1, 0, 0] vector in corresponding frame
            gt_v_ga = gt_v_ga_raw / torch.norm(gt_v_ga_raw)

            # check rotations by rotating samples back into body-frame and taking the norm
            # acc_b_check = SO3.Exp(self.features[idx, 0:3, :].transpose(0, 1)).transpose(1, 2) @ C_gamma @ acc_ga.transpose(0, 1).unsqueeze(2)

            # print(torch.norm(acc_b_check - acc.transpose(0, 1).unsqueeze(2)))

            # C_vu_base = filtering_utils.unit_vec_rodrigues(torch.Tensor([1, 0, 0]).view(1, 3), gt_v_ga.view(1, 3))

            # phi_base_vu = SO3.Log(C_vu_base).reshape(3)

            phi_g0_b = SO3.Log(C_g0_b).squeeze(2).transpose(0, 1)

            return (
                torch.cat((phi_g0_b, acc_ga), dim=0),
                gt_v_ga.view(1, 3),
                # torch.norm(gt_v),
                # torch.norm(gt_v_initial),
            )


# %%
