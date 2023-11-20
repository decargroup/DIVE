# %%
import numpy as np
import torch
import scipy.constants
import pymlg
from os import path as osp
import time

from pymlg.torch import SO3
from pymlg.torch import SE23
from filtering.ekf import ExtendedKalmanFilterTorch
from filtering.measurement_models import NullQuadrotorMeasurements, SyntheticVelocityUnitVector, VelocityUnitVector, SyntheticVelocityUnitVectorGravityAligned, VelocityVector
from filtering.process_models import NullOnUpdateCoupledIMU, CoupledIMUKinematicModel
from filtering import filtering_utils

from pymlg.numpy import SE23 as SE23_np

import random

class VelocityVectorRegressor:
    """
    A wrapper class from a quadrotor model that uses unconstrained 6DOF IMU kinematics and a set of pseudomeasurements to perform pose estimation
    """

    def __init__(self, args, datalength):
        sigma_gyro_ct = args.sigma_gyro_ct
        sigma_accel_ct = args.sigma_accel_ct
        sigma_gyro_bias_ct = args.sigma_gyro_bias_ct
        sigma_accel_bias_ct = args.sigma_accel_bias_ct

        Q_c = torch.eye(12, 12)
        Q_c[0:3, 0:3] *= sigma_gyro_ct**2
        Q_c[3:6, 3:6] *= sigma_accel_ct**2
        Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
        Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

        self.Q_c = Q_c

        # based on reference frame, make gravity vector +ve up or down
        if args.z_up_frame:
            g_a = torch.tensor([0, 0, -scipy.constants.g])
        else:
            g_a = torch.tensor([0, 0, scipy.constants.g])

        self.null_coupled_imu = CoupledIMUKinematicModel(Q_c, args.perturbation, g_a)
        self.null_quad_meas = VelocityVector(args, Q_c, g_a)

        # initialize filter
        self.filter = ExtendedKalmanFilterTorch(
            self.null_coupled_imu, self.null_quad_meas
        )

        # initialize overall logging vector
        self.logging_vec = torch.empty(datalength, 56, 1)
        self.idx = 0
        self.initialized = False
        self.args = args

        # initialize overall IMU sample vector (for net feed-in)
        self.rmi_logger = torch.empty(7, 0)

        # initialize rmi buffer length and marker 
        self.rmi_marker = 0
        self.buffer_initialized = False

        # initialize RMI parameters based off input arguments
        imu_dt_bar = (1 / args.nominal_imu_frequency)

        self.discrete_update_step = int((1 / args.update_frequency) / imu_dt_bar)
        self.rmi_logging_range = int(args.inertial_window_length / imu_dt_bar)

        self.perform_update = False
        self.did_update = False
        if (self.args.reinitialize_after_inertial_window):
            self.first_update = True
        else:
            self.first_update = False

    def initialize(self, P_0, C_0, v_0, r_0, b_g_0 = torch.zeros(1, 3, 1), b_a_0 = torch.zeros(1, 3, 1)):
        # for initialization. for now, set initial bias estimate to zero
        X_0 = SE23.from_components(C_0, v_0, r_0)
        self.x = list((X_0, torch.cat((b_g_0, b_a_0), dim=1)))

        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

        self.P = P_0

        # reset preintegration process model and initialize with current covariance
        self.null_coupled_imu.reset_incremental_jacobians(self.P)

        self.initialized = True

    def add_rmi(self, rmi, t_k):
        """
        accepts an rmi as a torch.Tensor of shape [2, 3]
        """

        # if logging size is larger than predefined range, then remove first element and append new element
        if self.rmi_logger.shape[1] > self.rmi_logging_range:
            self.rmi_logger = self.rmi_logger[:, 1:]

        rmi = torch.cat((rmi.reshape(6, 1), torch.Tensor([t_k]).view(1, 1)), dim=0)

        self.rmi_logger = torch.cat((self.rmi_logger, rmi), dim=1)

        # post-append, increment marker by 1 to indicate a new element
        self.rmi_marker += 1

        if (self.rmi_marker >= self.rmi_logging_range) and (not self.buffer_initialized):
            self.rmi_marker = 0
            self.buffer_initialized = True
            self.perform_update = True
        elif (self.buffer_initialized):
            if (self.rmi_marker >= self.discrete_update_step):
                self.rmi_marker = 0
                self.perform_update = True

    # helper for computing full history of rotation matricies from state vector
    def compute_rot_history(self):
        return SO3.Exp(self.x_history[:, 0:3])

    def de_bias_imu(self, u : torch.Tensor):

        # subtract current bias estimate from measurement
        u[:, 0, :] = u[:, 0, :] - self.x[1][:, :3].squeeze(2)
        u[:, 1, :] = u[:, 1, :] - self.x[1][:, 3:].squeeze(2)

        return u

    def predict(self, u, dt):
        """
        u is the RMI, which is a torch.Tensor of shape [2, 3]
        """

        u_db = self.de_bias_imu(u.clone().detach())

        self.x[0], self.P = self.filter.predict(self.x[0], self.P, u_db, dt)

        # collect into aggregate x for correction and/or logging
        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

    def correct(self, u, t_k, gt_k, dt):
        """
        u is the RMI, which is a torch.Tensor of shape [2, 3]
        """

        u_db = self.de_bias_imu(u.clone().detach())

        self.add_rmi(u_db, t_k)

        if self.perform_update:
            if (self.first_update):

                # if resetting bias as well, then retrieve from ground-truth instance
                if (self.args.contains_bias):
                    
                    # initialize initial covariance - for now, just make it equal to identity
                    P_0 = torch.eye(15, 15)
                    P_0[0:3, 0:3] *= self.args.sigma_theta_rp_init**2
                    P_0[3:6, 3:6] *= self.args.sigma_velocity_init**2
                    P_0[6:9, 6:9] *= self.args.sigma_position_init**2
                    P_0[9:12, 9:12] *= self.args.sigma_bias_gyro**2
                    P_0[12:15, 12:15] *= self.args.sigma_bias_acc**2

                    self.P = P_0.unsqueeze(0)

                    self.null_coupled_imu.reset_incremental_jacobians(self.P)

                    # pull out ground-truth values from gt_k
                    gt_phi = gt_k[:, 0]
                    gt_v = gt_k[:, 1]
                    gt_r = gt_k[:, 2]
                    gt_b_g = gt_k[:, 3]
                    gt_b_a = gt_k[:, 4]

                    X_extended_pose_true = SE23.from_components(SO3.Exp(gt_phi), gt_v.view(1, 3, 1), gt_r.view(1, 3, 1))

                    self.x = list((X_extended_pose_true, torch.cat((gt_b_g.view(1, 3, 1), gt_b_a.view(1, 3, 1)), dim=1)))

                    self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

                    print("Re-initializing filter with state (phi, v, r): ", gt_phi, gt_v, gt_r)
                    self.first_update = False
                    self.perform_update = False
                    self.did_update = False

                # otherwise, keep bias at currently estimated value
                else:
                    # initialize initial covariance - for now, just make it equal to identity
                    P_0 = torch.eye(15, 15)
                    P_0[0:3, 0:3] *= self.args.sigma_theta_rp_init**2
                    P_0[3:6, 3:6] *= self.args.sigma_velocity_init**2
                    P_0[6:9, 6:9] *= self.args.sigma_position_init**2
                    P_0[9:12, 9:12] *= self.args.sigma_bias_gyro**2
                    P_0[12:15, 12:15] *= self.args.sigma_bias_acc**2

                    self.P = P_0.unsqueeze(0)

                    self.null_coupled_imu.reset_incremental_jacobians(self.P)

                    # pull out ground-truth values from gt_k
                    gt_phi = gt_k[:, 0]
                    gt_v = gt_k[:, 1]
                    gt_r = gt_k[:, 2]

                    X_extended_pose_true = SE23.from_components(SO3.Exp(gt_phi), gt_v.view(1, 3, 1), gt_r.view(1, 3, 1))

                    self.x = list((X_extended_pose_true, self.x[1]))

                    self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

                    print("Re-initializing filter with state (phi, v, r): ", gt_phi, gt_v, gt_r)
                    self.first_update = False
                    self.perform_update = False
                    self.did_update = False
            else:
                x_hat, p_hat, self.did_update = self.filter.correct(self.rmi_logger, self.x, self.null_coupled_imu.P_j, dt)

                self.null_coupled_imu.reset_incremental_jacobians(p_hat)

                self.x = x_hat
                self.P = p_hat

                self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

                # after completing update, set perform_update to false
                self.perform_update = False
                self.did_update = True

class VelocityUnitVectorRegressor:
    """
    A wrapper class from a quadrotor model that uses unconstrained 6DOF IMU kinematics and a set of pseudomeasurements to perform pose estimation
    """

    def __init__(self, args, datalength):
        sigma_gyro_ct = args.sigma_gyro_ct
        sigma_accel_ct = args.sigma_accel_ct
        sigma_gyro_bias_ct = args.sigma_gyro_bias_ct
        sigma_accel_bias_ct = args.sigma_accel_bias_ct

        Q_c = torch.eye(12, 12)
        Q_c[0:3, 0:3] *= sigma_gyro_ct**2
        Q_c[3:6, 3:6] *= sigma_accel_ct**2
        Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
        Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

        # based on reference frame, make gravity vector +ve up or down
        if args.z_up_frame:
            g_a = torch.tensor([0, 0, -scipy.constants.g])
        else:
            g_a = torch.tensor([0, 0, scipy.constants.g])

        self.null_coupled_imu = CoupledIMUKinematicModel(Q_c, args.perturbation, g_a)
        self.null_quad_meas = VelocityUnitVector(args, Q_c, g_a)

        # initialize filter
        self.filter = ExtendedKalmanFilterTorch(
            self.null_coupled_imu, self.null_quad_meas
        )

        # initialize overall logging vector
        self.logging_vec = torch.empty(datalength, 56, 1)
        self.idx = 0
        self.initialized = False
        self.args = args

        # initialize overall IMU sample vector (for net feed-in)
        self.rmi_logger = torch.empty(7, 0)

        # initialize rmi buffer length and marker 
        self.rmi_marker = 0
        self.buffer_initialized = False

        # initialize RMI parameters based off input arguments
        imu_dt_bar = (1 / args.nominal_imu_frequency)

        self.discrete_update_step = int((1 / args.update_frequency) / imu_dt_bar)
        self.rmi_logging_range = int(args.inertial_window_length / imu_dt_bar)

        self.perform_update = False
        self.did_update = False
        if (self.args.reinitialize_after_inertial_window):
            self.first_update = True
        else:
            self.first_update = False

    def initialize(self, P_0, C_0, v_0, r_0, b_g_0 = torch.zeros(1, 3, 1), b_a_0 = torch.zeros(1, 3, 1)):
        # for initialization. for now, set initial bias estimate to zero
        X_0 = SE23.from_components(C_0, v_0, r_0)
        self.x = list((X_0, torch.cat((b_g_0, b_a_0), dim=1)))

        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

        self.P = P_0

        self.initialized = True

    def add_rmi(self, rmi, t_k):
        """
        accepts an rmi as a torch.Tensor of shape [2, 3]
        """

        # if logging size is larger than predefined range, then remove first element and append new element
        if self.rmi_logger.shape[1] > self.rmi_logging_range:
            self.rmi_logger = self.rmi_logger[:, 1:]

        rmi = torch.cat((rmi.reshape(6, 1), torch.Tensor([t_k]).view(1, 1)), dim=0)

        self.rmi_logger = torch.cat((self.rmi_logger, rmi), dim=1)

        # post-append, increment marker by 1 to indicate a new element
        self.rmi_marker += 1

        if (self.rmi_marker >= self.rmi_logging_range) and (not self.buffer_initialized):
            self.rmi_marker = 0
            self.buffer_initialized = True
            self.perform_update = True
        elif (self.buffer_initialized):
            if (self.rmi_marker >= self.discrete_update_step):
                self.rmi_marker = 0
                self.perform_update = True

    # helper for computing full history of rotation matricies from state vector
    def compute_rot_history(self):
        return SO3.Exp(self.x_history[:, 0:3])

    def de_bias_imu(self, u : torch.Tensor):

        # subtract current bias estimate from measurement
        u[:, 0, :] = u[:, 0, :] - self.x[1][:, :3].squeeze(2)
        u[:, 1, :] = u[:, 1, :] - self.x[1][:, 3:].squeeze(2)

        return u

    def predict(self, u, dt):
        """
        u is the RMI, which is a torch.Tensor of shape [2, 3]
        """
    
        u_db = self.de_bias_imu(u.clone().detach())

        self.x[0], self.P = self.filter.predict(self.x[0], self.P, u_db, dt)

        C, _, _ = SE23.to_components(self.x[0])
        # collect into aggregate x for correction and/or logging
        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

    def correct(self, u, t_k, gt_k, dt):
        """
        u is the RMI, which is a torch.Tensor of shape [2, 3]
        """

        self.add_rmi(u, t_k)

        if self.perform_update:
            if (self.first_update):
                
                # initialize initial covariance - for now, just make it equal to identity
                P_0 = torch.eye(15, 15)
                P_0[0:3, 0:3] *= self.args.sigma_theta_rp_init**2
                P_0[3:6, 3:6] *= self.args.sigma_velocity_init**2
                P_0[6:9, 6:9] *= self.args.sigma_position_init**2
                P_0[9:12, 9:12] *= self.args.sigma_bias_gyro**2
                P_0[12:15, 12:15] *= self.args.sigma_bias_acc**2

                self.P = P_0.unsqueeze(0)

                # pull out ground-truth values from gt_k
                gt_phi = gt_k[:, 0]
                gt_v = gt_k[:, 1]
                gt_r = gt_k[:, 2]

                X_extended_pose_true = SE23.from_components(SO3.Exp(gt_phi), gt_v.view(1, 3, 1), gt_r.view(1, 3, 1))

                X_k_true = list((X_extended_pose_true, torch.zeros(1, 6, 1)))

                self.agg_x = torch.cat((SE23.Log(X_k_true[0]), X_k_true[1]), dim=1)

                print("Re-initializing filter with state (phi, v, r): ", gt_phi, gt_v, gt_r)
                self.first_update = False
                self.perform_update = False
                self.did_update = False
            else:
                ksi_hat, p_hat, self.did_update = self.filter.correct(self.rmi_logger, self.agg_x, self.P, dt)

                self.agg_x = ksi_hat
                self.P = p_hat

                # after completing update, set perform_update to false
                self.perform_update = False
        
        self.x[0] = SE23.Exp(self.agg_x[:, :9, :])
        self.x[1] = self.agg_x[:, 9:, :]

def generate_random_rotation_on_great_circle(rad_err : float):
    lambda_2 = random.randrange(-rad_err * 1000, rad_err * 1000) / 1000
    sigma_2 = np.arccos(np.cos(rad_err) / np.cos(lambda_2))

    sigma_2 = -sigma_2 if (np.sign(lambda_2) == 1) else sigma_2

    phi = torch.Tensor([0, sigma_2, lambda_2]).view(1, 3, 1)

    return SO3.Exp(phi)

class SyntheticVelocityUnitVectorRegressor:
    """
    A wrapper class from a quadrotor model that uses unconstrained 6DOF IMU kinematics and a set of pseudomeasurements to perform pose estimation
    """

    def __init__(self, args, datalength):
        sigma_gyro_ct = args.sigma_gyro_ct
        sigma_accel_ct = args.sigma_accel_ct + .01
        sigma_gyro_bias_ct = args.sigma_gyro_bias_ct
        sigma_accel_bias_ct = args.sigma_accel_bias_ct

        Q_c = torch.eye(12, 12)
        Q_c[0:3, 0:3] *= sigma_gyro_ct**2
        Q_c[3:6, 3:6] *= sigma_accel_ct**2
        Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
        Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

        # based on reference frame, make gravity vector +ve up or down
        if args.z_up_frame:
            g_a = torch.tensor([0, 0, -scipy.constants.g])
        else:
            g_a = torch.tensor([0, 0, scipy.constants.g])

        self.null_coupled_imu = NullOnUpdateCoupledIMU(Q_c, args.perturbation, g_a)
        self.null_quad_meas = SyntheticVelocityUnitVectorGravityAligned(args, Q_c, g_a)

        # initialize filter
        self.filter = ExtendedKalmanFilterTorch(
            self.null_coupled_imu, self.null_quad_meas
        )

        # initialize overall logging vector
        self.logging_vec = torch.empty(datalength, 56, 1)
        self.z = torch.zeros(1, 3, 1)

        self.idx = 0

        self.initialized = False

        self.args = args

        self.marker = torch.zeros(5)

    def initialize(self, P_0, C_0, v_0, r_0):
        # for initialization. for now, set initial bias estimate to zero
        X_0 = SE23.from_components(C_0, v_0, r_0)
        self.x = list((X_0, torch.zeros(1, 6, 1)))

        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

        self.P = P_0

        self.initialized = True

    # helper for computing full history of rotation matricies from state vector
    def compute_rot_history(self):
        return SO3.Exp(self.x_history[:, 0:3])

    def de_bias_imu(self, u : torch.Tensor):

        # subtract current bias estimate from measurement
        u[:, 0, :] = u[:, 0, :] - self.x[1][:, :3].squeeze(2)
        u[:, 1, :] = u[:, 1, :] - self.x[1][:, 3:].squeeze(2)

        return u

    def predict(self, u, dt):

        u_db = self.de_bias_imu(u.clone().detach())

        self.x[0], self.P = self.filter.predict(self.x[0], self.P, u_db, dt)

        # collect into aggregate x for correction and/or logging
        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

    def correct(self, u, dt):

        gt_phi = u[:, :, 0]
        gt_v = u[:, :, 2]

        gt_C = SO3.Exp(gt_phi)
        # compute overall rotation for imu samples
        anchor_euler = SO3.to_euler(gt_C, order="123")
        C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))

        # compute measurement
        self.z = ((C_gamma.transpose(1, 2) @ gt_v.reshape(3, 1)) / torch.norm(gt_v, dim=1).reshape(1, 1)).view(1, 3, 1)

        # add random noise of aproximately 15 degrees to measurement
        if (self.args.add_noise):
            C_err = generate_random_rotation_on_great_circle(self.args.desired_radian_error_norm)

            self.z = C_err @ self.z

        ksi_hat, p_hat = self.filter.correct(self.z, self.agg_x, self.P, dt)

        self.agg_x = ksi_hat
        self.P = p_hat
        
        self.x[0] = SE23.Exp(self.agg_x[:, :9, :])
        self.x[1] = self.agg_x[:, 9:, :]


class SyntheticMotionClassifier:
    """
    A wrapper class from a quadrotor model that uses unconstrained 6DOF IMU kinematics and a set of pseudomeasurements to perform pose estimation
    """

    def __init__(self, args, datalength):
        sigma_gyro_ct = args.sigma_gyro_ct
        sigma_accel_ct = args.sigma_accel_ct + .01
        sigma_gyro_bias_ct = args.sigma_gyro_bias_ct
        sigma_accel_bias_ct = args.sigma_accel_bias_ct

        Q_c = torch.eye(12, 12)
        Q_c[0:3, 0:3] *= sigma_gyro_ct**2
        Q_c[3:6, 3:6] *= sigma_accel_ct**2
        Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
        Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

        # based on reference frame, make gravity vector +ve up or down
        if args.z_up_frame:
            g_a = torch.tensor([0, 0, -scipy.constants.g])
        else:
            g_a = torch.tensor([0, 0, scipy.constants.g])

        self.null_coupled_imu = NullOnUpdateCoupledIMU(Q_c, args.perturbation, g_a)
        self.null_quad_meas = NullQuadrotorMeasurements(args, Q_c, g_a)

        # initialize filter
        self.filter = ExtendedKalmanFilterTorch(
            self.null_coupled_imu, self.null_quad_meas
        )

        # initialize overall logging vector
        self.logging_vec = torch.empty(datalength, 56, 1)

        self.idx = 0

        self.initialized = False

        self.args = args

        self.marker = torch.zeros(5)

    def initialize(self, P_0, C_0, v_0, r_0):
        # for initialization. for now, set initial bias estimate to zero
        X_0 = SE23.from_components(C_0, v_0, r_0)
        self.x = list((X_0, torch.zeros(1, 6, 1)))

        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

        self.P = P_0

        self.initialized = True

    # helper for computing full history of rotation matricies from state vector
    def compute_rot_history(self):
        return SO3.Exp(self.x_history[:, 0:3])

    def update(self, y : torch.Tensor):
        # generate boolean marker to represent which desired motion profiles are currently true given the synthetic ground truth
        # y is expected to be a marker in the shape of (1, 3, 4)

        gt_phi = y[:, :, 0]
        gt_omega_b = y[:, :, 1]
        gt_v = y[:, :, 2]
        gt_r = y[:, :, 3]

        self.marker = torch.zeros(5)

        if (torch.linalg.norm(gt_omega_b) < self.args.zero_omega_epsilon):
            self.marker[0] = 0
        if (torch.linalg.norm(gt_v) < self.args.zero_velocity_epsilon):
            self.marker[1] = 1
        
        # if not-zero velocity, proceed with specific pseudomeasurments (in the body frame)
        else:
            gt_v_b = SO3.Exp(gt_phi).transpose(1, 2) @ gt_v.reshape(3, 1)

            # upwards velocity marker
            if (torch.abs(gt_v_b[:, 2]) < self.args.zero_upward_velocity_epsilon):
                self.marker[2] = 1
            
            # lateral velocity marker
            if (torch.abs(gt_v_b[:, 1]) < self.args.zero_upward_velocity_epsilon):
                self.marker[3] = 1

            # forward velocity marker
            if (torch.abs(gt_v_b[:, 0]) < self.args.zero_upward_velocity_epsilon):
                self.marker[4] = 1

        # ensure that the null velocity and null planar velocity markers are not active at the same time
        if (self.marker[1] and (self.marker[2] or self.marker[3])):
            raise RuntimeError("Null Velocity and Null non-forward velocity markers are active at the same time!")

        # pass the marker to the measurement and process models to form the corresponding jacobians

        self.null_quad_meas.marker = self.marker

    def de_bias_imu(self, u : torch.Tensor):

        # subtract current bias estimate from measurement
        u[:, 0, :] = u[:, 0, :] - self.x[1][:, :3].squeeze(2)
        u[:, 1, :] = u[:, 1, :] - self.x[1][:, 3:].squeeze(2)

        return u

    def predict(self, u, dt):

        u_db = self.de_bias_imu(u.clone().detach())

        self.x[0], self.P = self.filter.predict(self.x[0], self.P, u_db, dt)

        # collect into aggregate x for correction and/or logging
        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

    def correct(self, u, dt):

        # only perform correction if any marker is active
        if (torch.any(self.marker)):

            # based on marker, aggregate measurement
            omega = u[:, 0, :].unsqueeze(2)
            acc = u[:, 1, :].unsqueeze(2)

            self.null_quad_meas.assign_u(omega, acc)

            m = torch.empty(1, 0, 1)

            # if zero-angular velocity update, add angular velocity
            if (self.marker[0]):
                m = torch.cat((m, torch.zeros(1, 3, 1)), dim=1)

            # if zero-velocity update, add null pseudomeasurement and acceleration
            if (self.marker[1]):
                
                m = torch.cat((m, torch.zeros(1, 3, 1)), dim=1)
                m = torch.cat((m, torch.zeros(1, 3, 1)), dim=1)

            else:
                # if planar-velocity update, add null pseudomeasurement
                if (self.marker[2]):
                    m = torch.cat((m, torch.zeros(1, 1, 1)), dim=1)
                # if lateral-velocity update, add null pseudomeasurement
                if (self.marker[3]):
                    m = torch.cat((m, torch.zeros(1, 1, 1)), dim=1)
                # if forward-velocity update, add null pseudomeasurement
                if (self.marker[4]):
                    m = torch.cat((m, torch.zeros(1, 1, 1)), dim=1)

            ksi_hat, p_hat = self.filter.correct(m, self.agg_x, self.P, dt)

            self.agg_x = ksi_hat
            self.P = p_hat
            
            self.x[0] = SE23.Exp(self.agg_x[:, :9, :])
            self.x[1] = self.agg_x[:, 9:, :]


# %%
