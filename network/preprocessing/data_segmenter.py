# %%
import torch
import numpy as np

from data import hdf5_loader

def segment_imu_data(measurements : torch.Tensor, N : float, freq : float, dt : float):
    """
    Returns segments of IMU measurement data based on the desired inertial window and discretization rate

    Parameters
    ----------
    measurements : torch.Tensor
        tensor containing entire measurement history with shape [N, 12]
    N : float
        desired length of inertial window in seconds
    freq : float 
        desired discretization rate in Hz. for example, for an N = 1s and freq = 20 Hz, a 1 second window of data will be sampled at 20 Hz
    dt : float
        the rate of the IMU measurements

    Returns
    -------
    segments : torch.Tensor
        a tensor representation of the segmented data, of shape [num_segments, 6, N_discrete_length]

    """

    # calculate window and stepping sizes
    window_index_length = int(N / dt)
    discrete_step = int((1 / freq) / dt)

    # unfold measurements to segment and declare requires_grad to be true
    segments = measurements.unfold(0, window_index_length, discrete_step)

    return segments

def segment_gt_data(gt_phi : torch.Tensor, gt_v : torch.Tensor, gt_r : torch.Tensor, N : float, freq : float, dt : float):
    """
    Returns segments of ground truth data based on the desired inertial window and discretization rate

    Parameters
    ----------
    gt_phi : torch.Tensor
        tensor containing entire orientation ground truth, at the IMU rate with shape [N, 3]
    gt_v : torch.Tensor
        tensor containing entire velocity ground truth, at the IMU rate with shape [N, 3]
    gt_r : torch.Tensor
        tensor containing entire position ground truth, at the IMU rate with shape [N, 3]
    N : float
        desired length of inertial window in seconds
    freq : float 
        desired discretization rate in Hz. for example, for an N = 1s and freq = 20 Hz, a 1 second window of data will be segmented at 20 Hz
    dt : float
        the rate of the IMU measurements

    Returns
    -------
    gt_segments : torch.Tensor
        a tensor representation of the segmented data, of shape [num_segments, 9, N_discrete_length]

    """

    window_index_length = int(N / dt)
    discrete_step = int((1 / freq) / dt)

    gt_phi_segments = gt_phi.unfold(0, window_index_length, discrete_step)
    gt_r_segments = gt_r.unfold(0, window_index_length, discrete_step)
    gt_v_segments = gt_v.unfold(0, window_index_length, discrete_step)

    gt_segments = torch.cat((gt_phi_segments, gt_v_segments, gt_r_segments), dim=1)

    return gt_segments

# %%
