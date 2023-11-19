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

from pymlg.torch import SO3, SE23

from scipy.spatial.transform import Rotation as R
from scipy import interpolate

import pymap3d as pm


def convert_to_ENU(lat, lon, alt):
    """converts lat, lon, alt to ENU coordinates"""
    x, y, z = pm.geodetic2enu(lat, lon, alt, lat[0], lon[0], alt[0])

    return x, y, z


class ThreeAxisUnivariateSpline:
    def __init__(self, ts, x, y, z):
        self.spline_x = interpolate.interp1d(ts, x, fill_value="extrapolate")
        self.spline_y = interpolate.interp1d(ts, y, fill_value="extrapolate")
        self.spline_z = interpolate.interp1d(ts, z, fill_value="extrapolate")

    def interpolate(self, target_ts):
        x_interp = self.spline_x(target_ts)
        y_interp = self.spline_y(target_ts)
        z_interp = self.spline_z(target_ts)

        return np.stack((x_interp, y_interp, z_interp), axis=1)


class FourAxisUnivariateSpline:
    def __init__(self, ts, x, y, z, w):
        self.spline_x = interpolate.interp1d(ts, x)
        self.spline_y = interpolate.interp1d(ts, y)
        self.spline_z = interpolate.interp1d(ts, z)
        self.spline_w = interpolate.interp1d(ts, w)

    def interpolate(self, target_ts):
        x_interp = self.spline_x(target_ts)
        y_interp = self.spline_y(target_ts)
        z_interp = self.spline_z(target_ts)
        w_interp = self.spline_w(target_ts)

        return np.stack((x_interp, y_interp, z_interp, w_interp), axis=1)


def load_cloud_dataset(args):
    # generate file location based on arguments
    f = os.path.join(args.root_dir, args.data_dir, args.file_target)

    # retrieve individual CSVs and np representations
    ds_quat = np.genfromtxt(os.path.join(f, "attitude.csv"), delimiter=",")[1:, :]
    ds_gps = np.genfromtxt(os.path.join(f, "gps.csv"), delimiter=",")[1:, :]
    ds_vel = np.genfromtxt(os.path.join(f, "velocities.csv"), delimiter=",")[1:, :]
    ds_imu = np.genfromtxt(os.path.join(f, "imu.csv"), delimiter=",")[
        args.start_idx :, :
    ]

    # generate valid time range to restrict data
    imu_ts = ds_imu[:, 0]
    vel_ts = ds_vel[:, 0]
    quat_ts = ds_quat[:, 0]

    lower_bound_ts = np.max([vel_ts[0], quat_ts[0]])
    upper_bound_ts = np.min([vel_ts[-1], quat_ts[-1]])

    # restrict data to valid time range
    valid_imu_ts_indicies = np.where(
        np.logical_and(imu_ts >= lower_bound_ts, imu_ts <= upper_bound_ts)
    )[0]

    # initialize individual splines
    quat_spline = FourAxisUnivariateSpline(
        quat_ts, ds_quat[:, 1], ds_quat[:, 2], ds_quat[:, 3], ds_quat[:, 4]
    )
    vel_spline = ThreeAxisUnivariateSpline(
        vel_ts, ds_vel[:, 1], ds_vel[:, 2], ds_vel[:, 3]
    )

    # interpolate splines to IMU frequency
    quat_interp = quat_spline.interpolate(imu_ts[valid_imu_ts_indicies])
    vel_interp = vel_spline.interpolate(imu_ts[valid_imu_ts_indicies])

    gyro = ds_imu[valid_imu_ts_indicies, 4:]
    acc = ds_imu[valid_imu_ts_indicies, 1:4]

    # batchify and normalize quaternions
    quat_interp = torch.Tensor(quat_interp)
    quat_interp /= torch.linalg.norm(quat_interp, dim=1)[:, None]

    gt_C = SO3.from_quat(torch.Tensor(quat_interp), ordering="xyzw")

    # generate relative timestamps in seconds
    ts = imu_ts[valid_imu_ts_indicies] - imu_ts[valid_imu_ts_indicies][0]
    ts = ts / 1e9

    # generate ENU coordinates for gt position
    lat = ds_gps[:, 1]
    lon = ds_gps[:, 2]
    alt = ds_gps[:, 3]

    x, y, z = convert_to_ENU(lat, lon, alt)

    pos_spline = ThreeAxisUnivariateSpline(ds_gps[:, 0], x, y, z)
    pos_interp = pos_spline.interpolate(imu_ts[valid_imu_ts_indicies])

    # calculating raw gps time relative to starting IMU time
    valid_gps_indicies = np.where(
        np.logical_and(ds_gps[:, 0] >= lower_bound_ts, ds_gps[:, 0] <= upper_bound_ts)
    )[0]
    raw_gps_ts = ds_gps[valid_gps_indicies, 0] - imu_ts[valid_imu_ts_indicies][0]
    raw_gps_ts = raw_gps_ts / 1e9

    trajectory_dict = {
        "C": gt_C,
        "v": torch.Tensor(vel_interp),
        "r": torch.Tensor(pos_interp),
        "gyro": torch.Tensor(gyro).squeeze(0),
        "gt_omega_b": torch.Tensor(gyro).squeeze(0),
        "acc": torch.Tensor(acc).squeeze(0),
        "ts": torch.Tensor(ts),
        "gt_q": torch.Tensor(quat_interp),
        "raw_gps": torch.Tensor(
            np.concatenate(
                (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1
            )
        ),
        "raw_gps_ts": torch.Tensor(raw_gps_ts),
    }

    return (f, trajectory_dict)


def load_cloud_dataset_interpolated(args):
    f, traj_dict = load_cloud_dataset(args)


class CloudTrajectoryDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

    def __len__(self):
        # by default, currently only loads one long-form trajectory at a time, so return 1
        return 1

    def __getitem__(self, idx):
        return load_cloud_dataset(self.args)


def load_euroc(f, args):
    # retrieve array representations of all ground truth variables
    gt = np.genfromtxt(
        os.path.join(f, "state_groundtruth_estimate0", "data.csv"), delimiter=","
    )[args.start_idx :, :]
    # retrieve time representation in seconds relative to starting index
    ts = gt[:, 0] / 1e9

    # retrieve individual quat, pos, and vel representations
    gt_pos = gt[:, 1:4]
    gt_quat_wxyz = gt[:, 4:8]
    gt_vel = gt[:, 8:11]
    gt_b_g = gt[:, 11:14]
    gt_b_a = gt[:, 14:17]

    # retrieve imu measurements
    imu = np.genfromtxt(os.path.join(f, "imu0", "data.csv"), delimiter=",")[
        args.start_idx :, :
    ]

    # retrieve corresponding timestamps and measurements
    imu_ts = imu[:, 0] / 1e9
    imu_gyro = imu[:, 1:4]
    imu_acc = imu[:, 4:7]

    # the IMU frame must be transformed into the gravity-up inertial frame
    C_bg_bi = torch.zeros(3, 3).unsqueeze(0)
    C_bg_bi[:, 0, 2] = 1
    C_bg_bi[:, 1, 1] = -1
    C_bg_bi[:, 2, 0] = 1

    # correct bias representations
    gt_b_g_corrected = (C_bg_bi @ torch.Tensor(gt_b_g).unsqueeze(2)).squeeze(2).numpy()
    gt_b_a_corrected = (C_bg_bi @ torch.Tensor(gt_b_a).unsqueeze(2)).squeeze(2).numpy()

    imu_acc_corrected = (
        (C_bg_bi @ torch.Tensor(imu_acc).unsqueeze(2)).squeeze(2).numpy()
    )
    imu_gyro_corrected = (
        (C_bg_bi @ torch.Tensor(imu_gyro).unsqueeze(2)).squeeze(2).numpy()
    )

    # initialize individual splines
    quat_spline = FourAxisUnivariateSpline(
        ts,
        gt_quat_wxyz[:, 0],
        gt_quat_wxyz[:, 1],
        gt_quat_wxyz[:, 2],
        gt_quat_wxyz[:, 3],
    )
    vel_spline = ThreeAxisUnivariateSpline(ts, gt_vel[:, 0], gt_vel[:, 1], gt_vel[:, 2])
    pos_spline = ThreeAxisUnivariateSpline(ts, gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2])
    b_g_spline = ThreeAxisUnivariateSpline(
        ts, gt_b_g_corrected[:, 0], gt_b_g_corrected[:, 1], gt_b_g_corrected[:, 2]
    )
    b_a_spline = ThreeAxisUnivariateSpline(
        ts, gt_b_a_corrected[:, 0], gt_b_a_corrected[:, 1], gt_b_a_corrected[:, 2]
    )

    gyro_spline = ThreeAxisUnivariateSpline(
        imu_ts,
        imu_gyro_corrected[:, 0],
        imu_gyro_corrected[:, 1],
        imu_gyro_corrected[:, 2],
    )
    acc_spline = ThreeAxisUnivariateSpline(
        imu_ts,
        imu_acc_corrected[:, 0],
        imu_acc_corrected[:, 1],
        imu_acc_corrected[:, 2],
    )

    # find valid time range
    lower_bound_ts = np.max([ts[0], imu_ts[0]])
    upper_bound_ts = np.min([ts[-1], imu_ts[-1]])

    imu_ts_frequency_corrected = torch.arange(
        start=lower_bound_ts, end=upper_bound_ts, step=(1 / 400)
    ).numpy()

    # interpolate splines to IMU frequency
    quat_interp = quat_spline.interpolate(imu_ts_frequency_corrected)
    vel_interp = vel_spline.interpolate(imu_ts_frequency_corrected)
    pos_interp = pos_spline.interpolate(imu_ts_frequency_corrected)
    b_g_interp = b_g_spline.interpolate(imu_ts_frequency_corrected)
    b_a_interp = b_a_spline.interpolate(imu_ts_frequency_corrected)
    gyro_interp = gyro_spline.interpolate(imu_ts_frequency_corrected)
    acc_interp = acc_spline.interpolate(imu_ts_frequency_corrected)

    # batchify and normalize quaternions
    quat_interp = torch.Tensor(quat_interp)
    quat_interp /= torch.linalg.norm(quat_interp, dim=1)[:, None]

    C_a_bi = SO3.from_quat(torch.Tensor(quat_interp), ordering="wxyz")

    # move ground-truth values to x-forward frame
    C_a_bg = C_a_bi @ C_bg_bi.transpose(1, 2)
    v = torch.Tensor(vel_interp).view(-1, 3, 1)
    r = torch.Tensor(pos_interp).view(-1, 3, 1)
    b_g = torch.Tensor(b_g_interp).view(-1, 3, 1)
    b_a = torch.Tensor(b_a_interp).view(-1, 3, 1)

    imu_ts_frequency_corrected = (
        imu_ts_frequency_corrected - imu_ts_frequency_corrected[0]
    )
    trajectory_dict = {
        "C": C_a_bg,
        "v": v.view(-1, 3),
        "r": r.view(-1, 3),
        "b_g": b_g.view(-1, 3),
        "b_a": b_a.view(-1, 3),
        "gyro": torch.Tensor(gyro_interp).view(-1, 3),
        "acc": torch.Tensor(acc_interp).view(-1, 3),
        "ts": torch.Tensor(imu_ts_frequency_corrected).view(-1),
    }

    return trajectory_dict


class EuRoCDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append((args.file_target))

        elif args.data_list_loc is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_list_loc)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(self.args.root_dir, self.args.data_dir, self.filenames[idx])

        return (f, load_euroc(f, self.args))

def load_dido_trajectory(f, args):
    # TODO: replace this with a generic dataloader so that different file sources can be used
    gt_dict = hdf5_loader.retrieve_trajectory(f, args.start_idx)

    # based on reference frame, make gravity vector +ve up or down
    if args.z_up_frame:
        g_a = torch.tensor([0, 0, -scipy.constants.g])
    else:
        g_a = torch.tensor([0, 0, scipy.constants.g])

    # preprocess IMU instances based on simulation target

    if args.use_gt and args.self_augment:
        raise RuntimeError(
            "Cannot use ground-truth IMU data and self-augment at the same time!"
        )

    if args.use_gt:
        gt_C = SO3.from_quat(gt_dict["gt_q"])
        gt_acc_b = (
            gt_C.transpose(1, 2) @ (gt_dict["gt_acc_a"] - g_a).unsqueeze(2)
        ).squeeze(2)

        trajectory_dict = {
            "C": gt_C,
            "v": gt_dict["gt_v"],
            "r": gt_dict["gt_r"],
            "gyro": gt_dict["gt_omega_b"],
            "gt_omega_b": gt_dict["gt_omega_b"],
            "acc": gt_acc_b,
            "ts": gt_dict["ts"],
        }

        return trajectory_dict

    elif args.self_augment:
        gt_C = SO3.from_quat(gt_dict["gt_q"])
        gt_acc_b = gt_C.transpose(1, 2) @ (gt_dict["gt_acc_a"] - g_a).unsqueeze(2)

        sigma_gyro_ct = args.sigma_gyro_ct
        sigma_accel_ct = args.sigma_accel_ct
        sigma_gyro_bias_ct = args.sigma_gyro_bias_ct
        sigma_accel_bias_ct = args.sigma_accel_bias_ct

        Q_c = torch.eye(12, 12)
        Q_c[0:3, 0:3] *= sigma_gyro_ct**2
        Q_c[3:6, 3:6] *= sigma_accel_ct**2
        Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
        Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

        # corrupt ground-truth IMU measurements
        imu_corruptor = imu_preprocessing.CorruptIMUTrajectory(Q_c)
        time = gt_dict["ts"][0]

        # declare empty measurements object
        measurements = torch.empty(0, 2, 3)

        for omega, acc, t_k in zip(gt_dict["gt_omega_b"], gt_acc_b, gt_dict["ts"]):
            # form delta from previous time
            dt = t_k - time
            time = t_k

            # generate corrupted measurement instance
            m_k = imu_corruptor.generate_instance(args, omega, acc, dt)

            measurements = torch.cat((measurements, m_k), dim=0)

        gyro = measurements[:, 0, :]
        acc = measurements[:, 1, :]

        trajectory_dict = {
            "C": gt_C,
            "v": gt_dict["gt_v"],
            "r": gt_dict["gt_r"],
            "gyro": gyro,
            "gt_omega_b": gt_dict["gt_omega_b"],
            "acc": acc,
            "ts": gt_dict["ts"],
        }

        return trajectory_dict

    # otherwise, if no groundtruth or augmentation specified, simply return raw IMU measurements from trajectory sample
    else:
        gt_C = SO3.from_quat(gt_dict["gt_q"], ordering="wxyz")
        trajectory_dict = {
            "C": gt_C,
            "v": gt_dict["gt_v"],
            "r": gt_dict["gt_r"],
            "gyro": gt_dict["gyro"],
            "gt_omega_b": gt_dict["gt_omega_b"],
            "gt_acc_a": gt_dict["gt_acc_a"],
            "acc": gt_dict["acc"],
            "ts": gt_dict["ts"],
        }

        return trajectory_dict

class DIDOTrajectoryDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append((args.file_target))

        elif args.data_list_loc is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_list_loc)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(
            self.args.root_dir,
            self.args.data_dir,
            self.filenames[idx],
            self.args.ground_truth_output_name,
        )

        # output reference directory
        f_out = osp.join(self.args.root_dir, self.args.data_dir, self.filenames[idx])

        return (f_out, load_dido_trajectory(f, self.args))


def temporary_padding_helper(tens: np.ndarray):
    """helper for padding tensors to be of same size"""

    tens = torch.Tensor(tens)

    tens_padded = tens

    if tens.ndim == 3:
        tens_padded = torch.nn.functional.pad(tens, (0, 0, 0, 0, 1, 0))

    # if tensor is not already 3D, make it so
    if tens.ndim == 2:
        tens = tens.unsqueeze(2)

        tens_padded = torch.nn.functional.pad(tens, (0, 0, 0, 0, 1, 0))

    if tens.ndim == 1:
        tens_padded = torch.nn.functional.pad(tens, (1, 0))

    tens_padded[0] = tens[0]

    return tens_padded


class TLIOEstimationDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append(osp.join(args.file_target))

        elif args.tlio_data_list is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_estimation_dir, args.tlio_data_list)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    @staticmethod
    def load_tlio_estimation_dataset(f, args):
        """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

        states = np.load(f)

        # boilerplate code for retrieving individual elements from overall evolving state
        Rs = states[:, :9].reshape(-1, 3, 3)
        rs = SO3.Log(torch.Tensor(Rs)).numpy()
        vs = states[:, 9:12]
        ps = states[:, 12:15]
        ba = states[:, 15:18]
        bg = states[:, 18:21]
        accs = states[:, 21:24]  # offline calib compensated, scale+bias
        gyrs = states[:, 24:27]  # offline calib compensated, scale+bias
        ts = states[:, 27]
        sigma_r = np.sqrt(states[:, 28:31])
        sigma_v = np.sqrt(states[:, 31:34])
        sigma_p = np.sqrt(states[:, 34:37])
        sigma_bg = np.sqrt(states[:, 37:40])
        sigma_ba = np.sqrt(states[:, 40:43])
        innos = states[:, 43:46]
        meas = states[:, 46:49]
        pred = states[:, 49:52]
        meas_sigma = states[:, 52:55]
        inno_sigma = states[:, 55:58]
        nobs_sigma = states[:, 58 : 58 + 16]

        trajectory_dict = {
            "C": temporary_padding_helper(Rs),
            "v": temporary_padding_helper(vs),
            "r": temporary_padding_helper(ps),
            "ts": temporary_padding_helper(ts),
            "sigma_phi": temporary_padding_helper(sigma_r),
            "sigma_v": temporary_padding_helper(sigma_v),
            "sigma_r": temporary_padding_helper(sigma_p),
        }

        return trajectory_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(
            self.args.root_estimation_dir,
            self.args.estimation_data_dir,
            self.filenames[idx],
            self.args.tlio_filter_output_name,
        )

        return (self.filenames[idx], self.load_tlio_estimation_dataset(f, self.args))


class BlackbirdDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append((args.file_target))

        elif args.data_list_loc is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_list_loc)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(self.args.root_dir, self.args.data_dir, self.filenames[idx])

        return (f, load_blackbird(f, self.args))


def load_blackbird(f, args):
    # generate ground truth np representation and imu_csv representation
    gt = np.genfromtxt(os.path.join(f, "groundTruthPoses.csv"), delimiter=",")[
        args.start_idx :, :
    ]
    imu_csv = pd.read_csv(os.path.join(f, "blackbird_slash_imu.csv"))

    # generate DCMs to rotate ground truth and IMU measurements into gravity-up frame (as model expects)
    # C_ni -> rotates from IMU frame into NED frame
    # C_en -> rotates from NED frame into ENU frame (expected model input)

    C_en = SO3.Exp(torch.Tensor([0, 0, torch.pi / 2])) @ SO3.Exp(
        torch.Tensor([0, torch.pi, 0])
    )
    C_ni = SO3.Exp(torch.Tensor([0, 0, torch.pi / 2]))

    # retrieve imu samples resolved in original body frame
    omega_b_np = np.stack(
        (
            imu_csv["x.1"].values.reshape(-1, 1),
            imu_csv["y.1"].values.reshape(-1, 1),
            imu_csv["z.1"].values.reshape(-1, 1),
        ),
        axis=1,
    )[args.start_idx:, :, :]
    omega_b = torch.Tensor(omega_b_np)
    acc_b_np = np.stack(
        (
            imu_csv["x.2"].values.reshape(-1, 1),
            imu_csv["y.2"].values.reshape(-1, 1),
            imu_csv["z.2"].values.reshape(-1, 1),
        ),
        axis=1,
    )[args.start_idx:, :, :]
    acc_b = torch.Tensor(acc_b_np)

    # change imu samples to be resolved in gravity-up frame
    acc_enu = (C_en @ C_ni @ acc_b).squeeze(2)
    omega_enu = (C_en @ C_ni @ omega_b).squeeze(2)

    # retrieve ground truth samples resolved in original body frame

    # retrieve ground-truth quaternion, normalize, and convert to orientation
    gt_q_norm = torch.Tensor(gt[:, 4:])
    gt_q_norm /= torch.norm(gt_q_norm, dim=1).view(-1, 1)

    # rotate ground-truth position into gravity-up frame (ENU)
    gt_r = torch.Tensor(gt[:, 1:4]).unsqueeze(2)
    gt_r_enu = (C_en @ gt_r).squeeze(2)

    # retrieve time representation in seconds relative to starting index
    gt_ts = (gt[:, 0]) / 1e6

    # retrieve imu timestamps

    # retrieve rosbag time references
    imu_ts = (imu_csv["rosbagTimestamp"].values[args.start_idx :]) / 1e9

    # find lower and upper bounds for timestamps
    # find valid time range
    lower_bound_ts = np.max([gt_ts[0], imu_ts[0]])
    upper_bound_ts = np.min([gt_ts[-1], imu_ts[-1]])

    # initialize individual splines
    quat_spline = FourAxisUnivariateSpline(
        gt_ts - lower_bound_ts,
        gt_q_norm[:, 0],
        gt_q_norm[:, 1],
        gt_q_norm[:, 2],
        gt_q_norm[:, 3],
    )
    pos_spline = ThreeAxisUnivariateSpline(
        gt_ts - lower_bound_ts, gt_r_enu[:, 0], gt_r_enu[:, 1], gt_r_enu[:, 2]
    )

    gyro_spline = ThreeAxisUnivariateSpline(
        imu_ts - lower_bound_ts, omega_enu[:, 0], omega_enu[:, 1], omega_enu[:, 2]
    )
    acc_spline = ThreeAxisUnivariateSpline(
        imu_ts - lower_bound_ts, acc_enu[:, 0], acc_enu[:, 1], acc_enu[:, 2]
    )

    imu_ts_frequency_corrected = torch.arange(
        start=0, end=upper_bound_ts - lower_bound_ts, step=(1 / 400)
    ).numpy()

    # interpolate splines to IMU frequency
    quat_interp = quat_spline.interpolate(imu_ts_frequency_corrected)
    pos_interp = pos_spline.interpolate(imu_ts_frequency_corrected)
    gyro_interp = gyro_spline.interpolate(imu_ts_frequency_corrected)
    acc_interp = acc_spline.interpolate(imu_ts_frequency_corrected)

    # batchify and normalize quaternions
    quat_interp = torch.Tensor(quat_interp)
    quat_interp /= torch.linalg.norm(quat_interp, dim=1)[:, None]

    # generate valid interpolated rotations from quaternions
    gt_C_ab = SO3.from_quat(quat_interp, ordering="wxyz")
    gt_C_ae_be = C_en @ gt_C_ab @ C_en.transpose(1, 2)

    r = torch.Tensor(pos_interp).view(-1, 3, 1)

    # compute biases given initial stationary period and assign to ground-truth values
    b_g = torch.mean(torch.Tensor(gyro_interp)[:int(1 * args.nominal_imu_frequency), :], dim=0).view(1, 3)
    b_a = torch.mean(torch.Tensor(acc_interp)[:int(1 * args.nominal_imu_frequency), :], dim=0).view(1, 3)
    b_a[:, 2] -= scipy.constants.g

    trajectory_dict = {
        "C": gt_C_ae_be,
        "v": torch.zeros(r.shape[0], 3),
        "r": r.view(-1, 3),
        "gyro": torch.Tensor(gyro_interp).view(-1, 3),
        "acc": torch.Tensor(acc_interp).view(-1, 3),
        "b_g": b_g.repeat(r.shape[0], 1),
        "b_a": torch.zeros(r.shape[0], 3), # b_a.repeat(r.shape[0], 1),
        "ts": torch.Tensor(imu_ts_frequency_corrected).view(-1),
    }

    return trajectory_dict

class VisualOdometryFailureComparisonDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append(args.file_target)

        elif args.data_list_loc is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_dir, args.data_list_loc)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    @staticmethod
    def load_vi_comparison_dataset(f):

        states = np.load(f)

        gt_r = states[:, :3]
        r_proposed = states[:, 3:6]
        r_dead_reckoned = states[:, 6:9]

        trajectory_dict = {
            "gt_r": gt_r,
            "r_proposed": r_proposed,
            "r_dead_reckoned": r_dead_reckoned,
        }

        return trajectory_dict
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(
            self.args.root_dir,
            self.args.data_dir,
            self.filenames[idx],
            self.args.filter_output_name,
        )

        return (self.filenames[idx], self.load_vi_comparison_dataset(f))
        
class GenericEstimationDataset31(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append(args.file_target)

        elif args.data_list_loc is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_dir, args.data_list_loc)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    @staticmethod
    def load_generic_estimation_dataset(f):
        """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

        states = np.load(f)

        ksi = states[:, :9]
        sigma_states = states[:, 15:]
        # retrieve components from SE23 vector representation
        Cs, vs, rs = SE23.to_components(SE23.Exp(torch.Tensor(ksi).unsqueeze(2)))
        # boilerplate code for retrieving individual elements from overall evolving state
        ba = states[:, 9:12]
        bg = states[:, 12:15]
        ts = states[:, 30]
        sigma_phi = np.sqrt(sigma_states[:, 0:3])
        sigma_v = np.sqrt(sigma_states[:, 3:6])
        sigma_r = np.sqrt(sigma_states[:, 6:9])
        sigma_bg = np.sqrt(sigma_states[:, 9:12])
        sigma_ba = np.sqrt(sigma_states[:, 12:15])

        markers = states[:, 31:35]
        omega = states[:, 35:38]
        acc = states[:, 38:41]
        z = states[:, -15:-12]
        innov = states[:, -15:]
        nis = states[:, -1:]

        trajectory_dict = {
            "C": Cs,
            "v": vs,
            "r": rs,
            "ts": ts,
            "omega": omega,
            "acc": acc,
            "sigma_phi": sigma_phi,
            "sigma_v": sigma_v,
            "sigma_r": sigma_r,
            "sigma_bg": sigma_bg,
            "sigma_ba": sigma_ba,
            "pseudo_markers": markers,
            "ba": ba,
            "bg": bg,
            "innov": innov,
            "nis": nis,
            "meas": z,
        }

        return trajectory_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(
            self.args.root_dir,
            self.args.data_dir,
            self.filenames[idx],
            self.args.filter_output_name,
        )

        return (self.filenames[idx], self.load_generic_estimation_dataset(f))    


class GenericVUEstimationDataset(Dataset):
    def __init__(self, args) -> Dataset:
        # TODO: setting this to be script-specific for now. if it's fine, make it system-wide later
        torch.set_default_dtype(torch.float64)

        self.args = args

        # if file_target is not None, then running in single-shot mode
        if args.file_target is not None:
            # generate target file location
            self.filenames = []
            self.filenames.append(args.file_target)

        elif args.data_list is not None:
            """assumes line-seperated entries for trajectory targets"""
            data_list_loc = osp.join(args.root_dir, args.data_dir, args.data_list)

            # retrieve trajectory paths from constructor file
            data_f = open(data_list_loc, "r")
            self.filenames = data_f.read().splitlines()

        else:
            raise RuntimeError(
                "No file or datalist specified for syn-pseudo simulation!"
            )

    @staticmethod
    def load_generic_estimation_dataset(f):
        """code for plotting error plots with corresponding covariance bounds for TLIO filtering output and corresponding gt"""

        states = np.load(f)

        ksi = states[:, :9]
        sigma_states = states[:, 15:30]
        # retrieve components from SE23 vector representation
        Cs, vs, rs = SE23.to_components(SE23.Exp(torch.Tensor(ksi).unsqueeze(2)))
        # boilerplate code for retrieving individual elements from overall evolving state
        bg = states[:, 9:12]
        ba = states[:, 12:15]
        ts = states[:, 30]
        sigma_phi = np.sqrt(sigma_states[:, 0:3])
        sigma_v = np.sqrt(sigma_states[:, 3:6])
        sigma_r = np.sqrt(sigma_states[:, 6:9])
        sigma_bg = np.sqrt(sigma_states[:, 9:12])
        sigma_ba = np.sqrt(sigma_states[:, 12:15])

        omega = states[:, 31:34]
        acc = states[:, 34:37]
        z = states[:, 37:40]
        update_marker = states[:, 40]
        innov = states[:, 41:56]

        trajectory_dict = {
            "C": Cs,
            "v": vs,
            "r": rs,
            "ts": ts,
            "omega": omega,
            "acc": acc,
            "sigma_phi": sigma_phi,
            "sigma_v": sigma_v,
            "sigma_r": sigma_r,
            "sigma_bg": sigma_bg,
            "sigma_ba": sigma_ba,
            "ba": ba,
            "bg": bg,
            "meas": z,
            "update_marker": update_marker,
            "innov": innov,
        }

        return trajectory_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = osp.join(
            self.args.root_dir,
            self.args.data_dir,
            self.filenames[idx],
            self.args.filter_output_name,
        )

        return (self.filenames[idx], self.load_generic_estimation_dataset(f))
