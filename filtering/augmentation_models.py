import torch
from pymlg.torch import utils

from .process_models import CoupledIMUKinematicModel

class CoupledIMUAugmentationModel:

    def __init__(self, Q_c):
        self.dof = 15
        self.covariance_dof = 12

        self.Q_c = Q_c

    def state_jacobian(self, state_jacobian_singular : torch.Tensor, m : int):
        
        if (m > 0):
            id = utils.batch_eye(1, self.dof * m, self.dof * m)
            r0 = torch.cat((id, torch.zeros(1, self.dof * m, self.dof)), dim=2)
            r12 = torch.cat((torch.zeros(1, self.dof, self.dof * m), state_jacobian_singular), dim=2)

            state_jacobian = torch.cat((r0, r12, r12), dim=1)
        else:
            state_jacobian = torch.cat((state_jacobian_singular, state_jacobian_singular), dim=1)

        return state_jacobian

    def covariance(self, input_jacobian_singular, m, dt):

        # discrete-time noise paramaterization
        Q_d = self.Q_c / dt

        # augmented input jacobian

        if (m > 0):
            input_jacobian = torch.cat((torch.zeros(1, self.dof * m, self.covariance_dof), input_jacobian_singular, input_jacobian_singular), dim=1)
        else:
            input_jacobian = torch.cat((input_jacobian_singular, input_jacobian_singular), dim=1)

        return input_jacobian @ Q_d @ input_jacobian.transpose(1, 2)