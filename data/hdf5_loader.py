# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


def retrieve_trajectory(filename, start_idx=0):
    """
    retrieves the ground truth trajectory from a given hdf5 file

    Parameters
    ----------
    f : str
        the absolute path to the hdf5 file
    start_idx : int
        the index to start the trajectory from

    Returns
    -------
    gt_dict : dict
        a dictionary containing the ground truth trajectory data. it's constituents and their corresponding shapes will be listed below
        gt_q : torch.Tensor
            the ground truth orientation quaternion, of shape [N, 4]
        gt_v : torch.Tensor
            the ground truth velocity, of shape [N, 3]
        gt_r : torch.Tensor
            the ground truth position, of shape [N, 3]
        gyro : torch.Tensor
            the raw IMU measurement of angular velocity in the body frame, of shape [N, 3]
        acc : torch.Tensor
            the raw IMU measurement of acceleration in the body frame, of shape [N, 3]
        gt_omega_b : torch.Tensor
            the ground truth angular velocity in the body frame, of shape [N, 3]
        gt_acc_a : torch.Tensor
            the ground truth acceleration in the inertial frame, of shape [N, 3]
        ts : torch.Tensor
            the timestamps of the measurements, of shape [N]
    """
    with h5py.File(filename, "r") as f:

        # make all timestamps relative
        ts_all = f["ts"][()][start_idx:]
        ts_all = ts_all - ts_all[0]

        gt_r = f["gt_p"][()][start_idx:]
        gt_q = f["gt_q"][()][start_idx:]
        gt_v = f["gt_v"][()][start_idx:]
        acc = f["acc"][()][start_idx:]
        gyro = f["gyr"][()][start_idx:]
        ts_coll = ts_all
        gt_gyro = f["gt_gyr"][()][start_idx:]
        gt_acc = f["gt_acc"][()][start_idx:]

        gt_dict = {
            "gt_q": torch.Tensor(gt_q),
            "gt_v": torch.Tensor(gt_v),
            "gt_r": torch.Tensor(gt_r),
            "gyro": torch.Tensor(gyro),
            "acc": torch.Tensor(acc),
            "gt_omega_b": torch.Tensor(gt_gyro),
            "gt_acc_a": torch.Tensor(gt_acc),
            "ts": torch.Tensor(ts_coll),
        }

        f.close()

    return gt_dict


def retrieve_trajectory_non_dict(filename, start_idx=0):
    f = h5py.File(filename, "r")

    # make all timestamps relative
    ts_all = f["ts"][()]
    ts_all = ts_all - ts_all[0]

    gt_r = f["gt_p"][()][start_idx:]
    gt_q = f["gt_q"][()][start_idx:]
    gt_v = f["gt_v"][()][start_idx:]
    gt_gyro = f["gt_gyr"][()][start_idx:]
    ts_coll = ts_all[start_idx:]

    return (
        (gt_q),
        (gt_v),
        (gt_r),
        (ts_coll),
        (gt_gyro),
    )


def retrieve_validation_trajectory(filename, start_idx=0):
    f = h5py.File(filename, "r")

    # make all timestamps relative
    ts_all = f["ts"][()]
    # print(ts_all.shape)
    ts_all = ts_all - ts_all[0]

    gt_r = f["vio_p"][()][start_idx:]
    gt_q = f["vio_q_wxyz"][()][start_idx:]
    gt_v = f["vio_v"][()][start_idx:]
    ts_coll = ts_all[start_idx:]

    return (
        (gt_q),
        (gt_v),
        (gt_r),
        (ts_coll),
    )


# %%
