# %%
import torch
import numpy as np
import scipy.constants
from pymlg.torch import SO3
import pymlg

from filtering.ekf import ExtendedKalmanFilterTorch
from filtering.process_models import CoupledIMUKinematicModel

from pymlg.torch import SE23, SO3

class IMUProcessModel:
    """
    Generic de-coupled IMU process model
    """

    def __init__(
        self,
        q_c,
        phi_0=torch.zeros((3, 1)),
        p_0=torch.zeros((15, 15)),
        v_0=torch.zeros((3, 1)),
        r_0=torch.zeros((3, 1)),
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # initialize state and covariance
        self.x = torch.zeros((9, 1))
        self.p = p_0

        # initialize kinematic variables
        self.phi = phi_0
        self.C = SO3.Exp_torch(self.phi)
        self.v = v_0
        self.r = r_0

        # initialize gravity vector
        self.g_a = g_a

        # initialize state and covariance history tensors
        self.x_traj = torch.zeros(9, 1)
        self.p_traj = torch.zeros(15, 15)
        self.C_traj = torch.zeros(3, 3)

        self.first = True

        # initialize continuous time noise
        self.q_c = q_c

    def evaluate(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Implementation of f(x) and FPF^T + BLB^T. propogates state x_{k-1} to x_{k} and associated covariance p_{k-1} to p_{k}

        Parameters
        ----------
        u : Input
            The input value. Expected to be a tensor with shape [2, 3]
        dt : float
            The time interval :math:`\Delta t` between the two states, as a float
        """

        # implementing f(x)

        # retrieve individual IMU measurements
        omega_k = u[0]
        acc_k = u[1]

        self.r = (
            self.r + dt * self.v + (dt**2 / 2) * (self.g_a + self.C @ acc_k.reshape(-1, 1))
        )

        self.v = self.v + dt * self.g_a + dt * self.C @ acc_k.reshape(-1, 1)

        self.C = self.C @ SO3.Exp_torch(dt * omega_k)

        # discrete-time covariance (should be a [12x12] tensor)
        q_n = self.q_c / dt

        # retrieve process jacobians
        F_k = self.state_jacobian(u, dt)
        B_k = self.noise_jacobian(u, dt)

        self.p = F_k @ self.p @ F_k.T + B_k @ q_n @ B_k.T

        # transform into state vector
        self.x[0:3] = SO3.Log_torch(self.C).reshape(-1, 1)
        self.x[3:6] = self.v
        self.x[6:9] = self.r

        if self.first:
            self.first = False
            self.p_traj = self.p.expand(1, -1, -1)
            self.x_traj = self.x.expand(1, -1, -1)
            self.C_traj = self.C.expand(1, -1, -1)
        else:
            self.p_traj = torch.cat((self.p_traj, self.p.expand(1, -1, -1)))
            self.x_traj = torch.cat((self.x_traj, self.x.expand(1, -1, -1)))
            self.C_traj = torch.cat((self.C_traj, self.C.expand(1, -1, -1)))

    def state_jacobian(self, u, dt):
        """
        Returns the left process jacobian wrt. the state
        """

        C_k = self.C

        # retrieve individual IMU measurements
        omega_k = u[0]
        acc_k = u[1]

        F_k = torch.eye(15, 15)

        F_k[6:9, 3:6] = torch.eye(3, 3) * dt
        F_k[3:6, 0:3] = dt * SO3.wedge_torch(-C_k @ acc_k)
        F_k[6:9, 0:3] = (dt**2 / 2) * SO3.wedge_torch(-C_k @ acc_k)
        F_k[0:3, 9:12] = (
            C_k
            @ SO3.Exp_torch(dt * omega_k)
            @ (dt * SO3.left_jacobian_torch((-dt * omega_k).reshape(1, -1)))
        )
        F_k[3:6, 12:15] = dt * C_k
        F_k[6:9, 12:15] = (dt**2 / 2) * C_k

        return F_k

    def noise_jacobian(self, u, dt):
        """
        Returns the left process jacobian wrt. the noise
        """
        B_k = torch.zeros(15, 12)

        C_k = self.C

        # retrieve individual IMU measurements
        omega_k = u[0]

        B_k[0:3, 0:3] = (
            C_k
            @ SO3.Exp_torch(dt * omega_k)
            @ (dt * SO3.left_jacobian_torch((-dt * omega_k).reshape(1, -1)))
        )
        B_k[3:6, 3:6] = dt * C_k
        B_k[6:9, 3:6] = (dt**2 / 2) * C_k
        B_k[9:15, 6:12] = torch.eye(6, 6)

        return B_k


class IMUKinematicState:
    """
    IMUKinematicState class instance that holds current ground truth state as well as time-varying biases and noise
    """

    def __init__(self, d_t, Q_c, gravity_up=True):
        """

        Parameters
        ----------
        d_t : float
            propogation timestep
        Q_c : torch.tensor
            continuous time PSD, where Q_n = Q_c / dt
        gravity_up : bool
            boolean indicating whether IMU is being used in a gravity-up reference frame

        """

        # propogation timestep
        self.d_t = d_t

        # decoupled kinematic variables (initialize entirely to zero)
        self.phi = torch.zeros(3, 1)
        self.C = SO3.Exp_torch(self.phi)
        self.v = torch.zeros(3, 1)
        self.r = torch.zeros(3, 1)

        # time-varying biases
        self.b_g = torch.zeros(1, 3)
        self.b_a = torch.zeros(1, 3)

        # associated noise vector, non-time varying
        self.Q_n = Q_c / d_t

        if gravity_up:
            self.g_a = torch.tensor([[0], [0], [-scipy.constants.g]])
        else:
            self.g_a = torch.tensor([[0], [0], [scipy.constants.g]])

        # internal elapsed time
        self.time = 0

        # boolean for initial pass
        self.first = True

        # overall tensor for gt_r
        self.gt_r = torch.zeros(1, 3, 1)
        self.gt_v = torch.zeros(1, 3, 1)
        self.gt_phi = torch.zeros(1, 3, 1)
        self.gt_C = self.C
        self.gt_inertial_acc = torch.empty(1, 3, 1)

    # ground-truth propogation
    def evaluate(self, omega, acc, q_k):
        """
        Propogates inertial state (pose, velocity, bias) forward with noise vector and generated IMU instance
        """
        # generic decoupled inertial navigation equations
        self.r = (
            self.r + self.d_t * self.v + (self.d_t**2 / 2) * (self.g_a + self.C @ acc)
        )
        self.v = self.v + self.d_t * self.g_a + self.d_t * (self.C @ acc)
        self.C = self.C @ SO3.Exp_torch(omega * self.d_t)

        self.phi = SO3.Log_torch(self.C)

        self.b_g += self.d_t * q_k[0][6:9]
        self.b_a += self.d_t * q_k[0][9:12]

        self.time += self.d_t

        inertial_acc = self.C @ acc

        self.gt_r = torch.cat((self.gt_r, self.r))
        self.gt_v = torch.cat((self.gt_v, self.v))
        self.gt_phi = torch.cat((self.gt_phi, self.phi.unsqueeze(2)))
        self.gt_C = torch.cat((self.gt_C, self.C))
        self.gt_inertial_acc = torch.cat((self.gt_inertial_acc, inertial_acc))

def generate_random_vec(N):
    """
    Generates a K-dimensional column vector (where N is of KxK dimensionality) where the diagonals of N represent the variance associated with each
    variable
    """
    return torch.normal(
        torch.zeros(1, N.shape[0]),
        torch.reshape(torch.diagonal(torch.sqrt(N)), (1, -1)),
    )


def generate_instance(x: IMUKinematicState) -> torch.tensor:
    """
    Generates an IMU measurement instance for a sinusoidal acceleration profile applied to a circular trajectory, and updates
    the IMUKinematicState
    """

    # generate random vector for noise addition and bias propogation
    q_k = generate_random_vec(x.Q_n)

    omega_bar = torch.tensor([[0.0, 0.0, 1.0]])
    acc_bar = torch.tensor([[0.0], [-5.0], [scipy.constants.g]])

    omega_corrupt = omega_bar + x.b_g + q_k[0][0:3]
    acc_corrupt = torch.reshape(acc_bar, (1, -1)) + x.b_a + q_k[0][3:6]

    x.evaluate(omega_bar, acc_bar, q_k)

    return torch.cat((omega_corrupt, torch.reshape(acc_corrupt, (1, -1)))).expand(
        1, -1, -1
    )

class DeadReckoner:
    """
    A wrapper class from a quadrotor model that uses unconstrained 6DOF IMU kinematics and a set of pseudomeasurements to perform pose estimation
    """

    def __init__(self, args):
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
            g_a = torch.tensor([[0], [0], [-scipy.constants.g]])
        else:
            g_a = torch.tensor([[0], [0], [scipy.constants.g]])

        self.null_coupled_imu = CoupledIMUKinematicModel(Q_c, g_a)

        # initialize filter
        self.filter = ExtendedKalmanFilterTorch(
            self.null_coupled_imu, None
        )

        # initialize overall logging vector
        self.logging_vec = torch.empty(0, 31, 1)

        self.initialized = False

        self.args = args

    def initialize(self, P_0, C_0, v_0, r_0):
        # for initialization. for now, set initial bias estimate to zero
        X_0 = SE23.from_components(C_0, v_0, r_0)
        self.x = list((X_0, torch.zeros(1, 6, 1)))

        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)

        self.P = P_0

        self.initialized = True

    def predict(self, u, dt):

        self.x[0], self.P = self.filter.predict(self.x[0], self.P, u, dt)

        # collect into aggregate x for correction
        self.agg_x = torch.cat((SE23.Log(self.x[0]), self.x[1]), dim=1)
# %%
