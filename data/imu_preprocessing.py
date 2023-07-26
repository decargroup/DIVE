import torch
import numpy as np
import scipy.constants
from pylie.torch import SO3

def generate_random_vec(N):
    """
    Generates a K-dimensional column vector (where N is of KxK dimensionality) where the diagonals of N represent the variance associated with each
    variable
    """
    return torch.normal(
        torch.zeros(1, N.shape[0]),
        torch.reshape(torch.diagonal(torch.sqrt(N)), (1, -1)),
    )

class CorruptIMUTrajectory:
    def __init__(self, Q_c):

        # continuous-time IMU parameters
        self.Q_c = Q_c

        # time-varying biases (TODO: currently initialized to zero at startup - should be some parameter for startup bias variance)
        self.b_g = torch.zeros(1, 3)
        self.b_a = torch.zeros(1, 3)

    def generate_instance(self, args, omega_bar : torch.Tensor, acc_bar: torch.Tensor, dt : float) -> torch.tensor:
        """
        corrupts a set of IMU measurements based on a set of continuous-time IMU noise parameters
        """

        # for initial iteration at zero-timestep, set dt to default
        if (dt == 0):
            dt = args.dt_bar

        # generate random vector for noise addition and bias propogation
        q_k = generate_random_vec(self.Q_c / dt)

        omega_corrupt = omega_bar + self.b_g + q_k[0][0:3]
        acc_corrupt = torch.reshape(acc_bar, (1, -1)) + self.b_a + q_k[0][3:6]

        self.b_g += dt * q_k[0][6:9]
        self.b_a += dt * q_k[0][9:12]

        return torch.cat((omega_corrupt, torch.reshape(acc_corrupt, (1, -1)))).expand(
            1, -1, -1
        )
    
def corrupt_measurement_history(imuTraj : CorruptIMUTrajectory, gt_gyro : torch.Tensor, gt_acc : torch.Tensor, ts : torch.Tensor):

    # capture initial time
    time = ts[0]

    # initialize measurements
    measurements = torch.empty(0, 2, 3)

    # iterate through history
    for omega, acc, t_k in zip(torch.Tensor(gt_gyro), gt_acc, ts[1:]):
        # form delta from previous time
        dt = t_k - time
        time = t_k

        # generate corrupted measurement instance
        m_k = imuTraj.generate_instance(omega, acc, dt)

        measurements = torch.cat((measurements, m_k), dim=0)

    return measurements

def rotate_gt_acc(gt_acc_a : torch.Tensor, gt_q : torch.Tensor):

    # complete in-place conversion
    gt_c = SO3.from_quat((gt_q), ordering="wxyz")

    gt_acc_b = gt_c.transpose(1, 2) @ ((gt_acc_a).reshape(-1, 3) - torch.tensor([0, 0, -scipy.constants.g])).reshape(-1, 3, 1)

    return gt_acc_b

def rotate_to_inertial(gt_3, gt_q):
    
    # complete in-place conversion
    gt_c = SO3.from_quat(gt_q)

    gt_acc_b = gt_c @ (gt_3).reshape(-1, 3, 1)

    return gt_acc_b

def rectangular_to_spherical(tens : torch.Tensor):
    """
    
    """
    pass