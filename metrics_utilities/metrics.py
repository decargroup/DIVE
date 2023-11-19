import numpy as np
import torch

import torch
from torch.utils.data import DataLoader

from pymlg.torch import SO3, SE23

from filtering import filtering_utils


def calculate_abs_rot_err(gt_C: np.ndarray, est_C: np.ndarray):
    """

    returns the absolute rotational error for a single trajectory with corresponding ground truth. expects both gt_C and est_C to have
    shape [N, 3, 3]

    """

    e_C = filtering_utils.calculate_rotation_error(gt_C, est_C)

    err_norm = np.linalg.norm(e_C, axis=1)

    are = np.sqrt(np.sum(err_norm**2) / gt_C.shape[0])

    return are


def calculate_abs_trans_err(gt_pos: np.ndarray, est_pos: np.ndarray):
    """

    returns the absolute translational error for a single trajectory
    with corresponding ground truth. expects both gt_pos and est_pos to have
    shape [N, 3, 1]

    """

    err_norm = np.linalg.norm((gt_pos - est_pos), axis=1)

    ate = np.sqrt(np.sum(err_norm**2) / gt_pos.shape[0])

    # print(gt_pos - est_pos)

    return ate


def calculate_relative_pose_err(
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    gt_v : np.ndarray,
    est_v : np.ndarray,
    gt_C: np.ndarray,
    est_C: np.ndarray,
    dt_bar: float,
    target_dt: float = 1.0,
):
    """

    returns the relative pose error for a single trajectory with corresponding ground truth. default target window is 1 second.
    both gt_pos and est_pos are expected to be [N, 3], and gt_C and est_C are expected to be [N, 3, 3]

    NOTE: could technically interpolate values here to make dt more specific, but for now, just using nominal delta
    since it should be relatively accurate over a large timespan (i.e. 1 second @ 400 Hz)
    """

    # convert to tensors for ease of batch operations
    gt_pos = torch.Tensor(gt_pos)
    est_pos = torch.Tensor(est_pos)
    gt_v = torch.Tensor(gt_v)
    est_v = torch.Tensor(est_v)
    gt_C = torch.Tensor(gt_C)
    est_C = torch.Tensor(est_C)

    T = SE23.from_components(gt_C, gt_v, gt_pos)

    T_hat = SE23.from_components(est_C, est_v, est_pos)

    # based on desired overall step, can exclude tail-end of estimated position and head of ground truth position
    delta_idx = int(target_dt / dt_bar)

    # index values to define relative errors
    T_i = T[:-delta_idx]
    T_delta = T[delta_idx:]

    T_hat_i = T_hat[:-delta_idx]
    T_delta_hat = T_hat[delta_idx:]

    F = SE23.inverse(SE23.inverse(T_i) @ T_delta) @ (SE23.inverse(T_hat_i) @ T_delta_hat)

    C_err, v_err, p_err = SE23.to_components(F)

    rpe_trans = torch.linalg.norm(p_err, dim=1) # torch.sqrt(torch.sum(torch.linalg.norm(p_err, dim=1)**2) / (gt_pos.shape[0] - delta_idx))
    rpe_vel = torch.linalg.norm(v_err, dim=1) # torch.sqrt(torch.sum(torch.linalg.norm(v_err, dim=1)**2) / (gt_pos.shape[0] - delta_idx))

    cos_angle = ((torch.vmap(torch.trace)(C_err) - 1) / 2).clamp(-1.0 + 1e-10, 1.0 - 1e-10)
    rpe_rot = torch.arccos(cos_angle) # torch.sum(torch.arccos(cos_angle)) / (gt_pos.shape[0] - delta_idx)
    
    return rpe_trans, rpe_vel, rpe_rot

def calculate_relative_trans_err(
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    gt_C: np.ndarray,
    est_C: np.ndarray,
    dt_bar: float,
    target_dt: float = 1.0,
):
    """

    returns the relative translational error for a single trajectory with corresponding ground truth. default target window is 1 second.
    both gt_pos and est_pos are expected to be [N, 3], and gt_C and est_C are expected to be [N, 3, 3]

    NOTE: could technically interpolate values here to make dt more specific, but for now, just using nominal delta
    since it should be relatively accurate over a large timespan (i.e. 1 second @ 400 Hz)
    """

    # convert to tensors for ease of batch operations
    gt_pos = torch.Tensor(gt_pos)
    est_pos = torch.Tensor(est_pos)
    gt_C = torch.Tensor(gt_C)
    est_C = torch.Tensor(est_C)

    gamma_aux = filtering_utils.c_3(SO3.Log(gt_C)) @ filtering_utils.c_3(
        SO3.Log(est_C)
    ).transpose(1, 2)

    # based on desired overall step, can exclude tail-end of estimated position and head of ground truth position
    delta_idx = int(target_dt / dt_bar)

    # index values to define relative errors
    indexed_err = gt_pos[delta_idx:] - est_pos[:-delta_idx]
    gamma_aux = gamma_aux[:-delta_idx]

    index_err_norm = torch.linalg.norm(indexed_err - gamma_aux @ (indexed_err), dim=1)

    rte = torch.sum(index_err_norm) / (gt_pos.shape[0] - delta_idx)

    return rte.numpy()


def retrieve_are_all(args, traj_dataloader: DataLoader, gt_dataloader: DataLoader):
    # overall are array
    are_ov = np.empty((0))

    for traj_tuple, gt_tuple in zip(traj_dataloader, gt_dataloader):
        f = traj_tuple[0][0]
        traj_dict = traj_tuple[1]

        gt_dict = gt_tuple[1]

        are_ov = np.concatenate(
            (are_ov, retrieve_are_single_instance(args, traj_dict, gt_dict)), axis=0
        )

    return are_ov


def retrieve_are_single_instance(args, states: dict, gt: dict):
    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0]
    rs = states["r"][0]
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)

    are = np.reshape(calculate_abs_rot_err(gt_C, Cs), (1))

    return are


def retrieve_ate_all(args, traj_dataloader: DataLoader, gt_dataloader: DataLoader):
    # overall ate array
    ate_ov = np.empty((0))

    for traj_tuple, gt_tuple in zip(traj_dataloader, gt_dataloader):
        f = traj_tuple[0][0]
        traj_dict = traj_tuple[1]

        gt_dict = gt_tuple[1]

        ate_ov = np.concatenate(
            (ate_ov, retrieve_ate_single_instance(args, traj_dict, gt_dict)), axis=0
        )

    return ate_ov


def retrieve_ate_single_instance(args, states: dict, gt: dict):
    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0]
    rs = states["r"][0]
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)

    ate = np.reshape(calculate_abs_trans_err(gt_r, rs), (1))

    return ate


def retrieve_rte_all(args, traj_dataloader: DataLoader, gt_dataloader: DataLoader):
    # overall ate array
    rte_ov = np.empty((0))

    for traj_tuple, gt_tuple in zip(traj_dataloader, gt_dataloader):
        f = traj_tuple[0][0]
        traj_dict = traj_tuple[1]

        gt_dict = gt_tuple[1]

        rte_ov = np.concatenate(
            (rte_ov, retrieve_rte_single_instance(args, traj_dict, gt_dict)), axis=0
        )

    return rte_ov


def retrieve_rte_single_instance(args, states: dict, gt: dict):
    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0]
    rs = states["r"][0]
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)

    rte = np.reshape(calculate_relative_trans_err(gt_r, rs, gt_C, Cs, 1 / 400, 1.), (1))

    return rte

def retrieve_rpe_all(args, traj_dataloader: DataLoader, gt_dataloader: DataLoader):
    # overall ate array
    rpe_trans_ov = np.empty((0))
    rpe_vel_ov = np.empty((0))
    rpe_rot_ov = np.empty((0))

    for traj_tuple, gt_tuple in zip(traj_dataloader, gt_dataloader):
        f = traj_tuple[0][0]
        traj_dict = traj_tuple[1]

        gt_dict = gt_tuple[1]

        rpe_trans, rpe_vel, rpe_rot = retrieve_rpe_single_instance(args, traj_dict, gt_dict)

        rpe_trans_ov = np.concatenate((rpe_trans_ov, rpe_trans))
        rpe_vel_ov = np.concatenate((rpe_vel_ov, rpe_vel))
        rpe_rot_ov = np.concatenate((rpe_rot_ov, rpe_rot))

    return rpe_trans_ov, rpe_vel_ov, rpe_rot_ov

def retrieve_rpe_single_instance(args, states: dict, gt: dict):
    # expand reference dictionaries for state estimate and ground truth. TODO: currently assuming single-batched state, make this more programmatic (?)
    Cs = states["C"][0]
    vs = states["v"][0]
    rs = states["r"][0]
    sigma_phi = states["sigma_phi"][0]
    sigma_v = states["sigma_v"][0]
    sigma_r = states["sigma_r"][0]
    ts = states["ts"][0]

    gt_C = gt["C"][0]
    gt_v = gt["v"][0].unsqueeze(2)
    gt_r = gt["r"][0].unsqueeze(2)

    # print(vs.shape)

    rpe_trans, rpe_vel, rpe_rot = calculate_relative_pose_err(gt_r, rs, gt_v, vs, gt_C, Cs, 1 / 400, 2)

    return rpe_trans.view(-1).numpy(), rpe_vel.view(-1).numpy(), rpe_rot.view(-1).numpy()
