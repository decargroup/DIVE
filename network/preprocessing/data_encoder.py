import torch
import numpy as np
from pymlg.torch import SE23, SO3
# helper to compute batched covariance
def batch_cov(T):
    B, N, D = T.size()
    mean = T.mean(dim=1).unsqueeze(1)
    diffs = (T - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)

# retrieve the velocity magnitude to mask low-velocity moments
def retrieve_velocity_magnitude(gt : torch.Tensor):
    """
    Encodes the ground truth kinematic values for binary classification. Note that all the boolean masks implicitly assume a gravity-aligned inertial frame

    Parameters
    ----------
    gt_segments : torch.Tensor
        tensor containing entire measurement history with shape [num_segments, 6, N_discrete_length]

    Returns
    -------
    segment_encoding : torch.Tensor
        a tensor boolean representation of the encoded data, of shape [num_segments]

    """

    # retrieve the last velocity vector of each segment
    gt_v_last = gt[:, :, -1]
    gt_v_first = gt[:, :, 0]

    # calculate the corresponding segment velocity magnitude
    return torch.norm(gt_v_first, dim=1), torch.norm(gt_v_last, dim=1)

# retrieve the ground truth velocity unit vector for a set of ground truth segments
def retrieve_velocity_unit_vector_encoding(gt : torch.Tensor):
    """
    Encodes the ground truth kinematic values for binary classification. Note that all the boolean masks implicitly assume a gravity-aligned inertial frame

    Parameters
    ----------
    gt_segments : torch.Tensor
        tensor containing entire measurement history with shape [num_segments, 3, N_discrete_length]

    Returns
    -------
    segment_encoding : torch.Tensor
        a tensor boolean representation of the encoded data, of shape [num_segments, 3]

    """

    # retrieve the last value rotation vector and velocity value of each segment
    # this yields a [num_segments, 6] tensor
    gt_phi_v_last = gt[:, :, -1]

    # retrieve the corresponding phi values
    gt_phi = gt_phi_v_last[:, 0:3]
    gt_v = gt_phi_v_last[:, 3:6]

    # calculate the body-frame velocity unit vector for each segment
    gt_vu_b = (SO3.Exp(gt_phi).transpose(1, 2) @ gt_v.unsqueeze(2)).squeeze(2) / torch.norm(gt_v, dim=1).unsqueeze(1)
    gt_vu_a = gt_v.reshape(-1, 3) / torch.norm(gt_v, dim=1).unsqueeze(1)

    return gt_vu_a

# assign labels to individual trajectories
def assign_planar_encoding(gt_segments : torch.Tensor, eps_eigval_min : float, eps_eigval_max : float, eps_xy_plane_thresh : float):
    """
    Encodes the ground truth kinematic values for binary classification. Note that all the boolean masks implicitly assume a gravity-aligned inertial frame

    Parameters
    ----------
    gt_segments : torch.Tensor
        tensor containing entire measurement history with shape [num_segments, 2, N_discrete_length, 3]
    eps_r : float
        position threshold for classification. if the "altitude" deviation from the beginning of the inertial window exceeds this, is labelled false.
    eps_v : float 
        velocity threshold for classification. if the velocity deviation from zero is less than this, is labelled as slow-moving and/or in hover and
        is not computed.
    eps_theta : float
        radian threshold for classification. if the trajectory does not have at least 1 inertial velocity vector with it's angle relative to the inertial XY
        plane below this threshold, it is labelled as non-planar. 

    Returns
    -------
    segment_encoding : torch.Tensor
        a tensor boolean representation of the encoded data, of shape [num_segments, 1]

    """

    # segment position in the z-axis and calculate difference relative to first value in segment
    gt_r_z = gt_segments[:, 0, :, 2]
    gt_r_z = gt_r_z - gt_r_z[:, 0].reshape(-1, 1)

    # calculate the batched covariance and eigendecomposition
    gt_r = gt_segments[:, 0, :, :]
    gt_r_cov = batch_cov(gt_r)
    cov_r_eigvals, cov_r_eigvecs = torch.linalg.eig(gt_r_cov)

    # cov_r_eigvals_min_mask = torch.any(torch.le(torch.real(cov_r_eigvals), eps_eigval_min), dim=1)
    cov_r_eigvals_max_mask = torch.any(torch.gt(torch.real(cov_r_eigvals), eps_eigval_max), dim=1)
    cov_r_eigvecs_xy_mask = torch.gt(torch.real(torch.abs(cov_r_eigvecs[:, 2, 2])), eps_xy_plane_thresh)

    planar_encoding = (cov_r_eigvals_max_mask * cov_r_eigvecs_xy_mask).long()

    # form encoding into vectorized form and declare that gradient needs to be tracked
    # planar_encoding = torch.stack((planar_encoding, planar_encoding.logical_not().float()), dim=1).reshape(-1, 2)
    # planar_encoding.requires_grad = True

    return planar_encoding, cov_r_eigvals, cov_r_eigvecs

def assign_zero_velocity_encoding(gt_segments : torch.Tensor, vel_norm_thresh : float = .01):
    """
    Encodes the ground truth kinematic values for binary classification. Note that all the boolean masks implicitly assume a gravity-aligned inertial frame

    Parameters
    ----------
    gt_segments : torch.Tensor
        tensor containing entire measurement history with shape [num_segments, 2, N_discrete_length, 3]
    vel_norm_thresh : float
        threshold for labelling an instance as zero-velocity

    Returns
    -------
    segment_encoding : torch.Tensor
        a tensor boolean representation of the encoded data, of shape [num_segments, 1]

    """

    # retrieve the last ground-truth velocity timestep of each segment
    gt_v_last = gt_segments[:, 1, -1, :]

    # calculate the corresponding norm and threshold them
    gt_v_last_norm = torch.linalg.norm(gt_v_last, dim=1)

    zero_vel_encoding = torch.le(gt_v_last_norm, vel_norm_thresh).long()

    return zero_vel_encoding