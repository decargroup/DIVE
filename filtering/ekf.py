# %%

import torch
import time
from .process_models import *

class ExtendedKalmanFilterTorch:
    """
    the rough beginning of an on-manifold kalman filter written in pytorch.
    """

    def __init__(self, process_model, measurement_model):

        self.process_model = process_model
        self.measurement_model = measurement_model

    def predict(self, x: torch.Tensor, p: torch.Tensor, u: torch.Tensor, dt: float):
        """
        propogates the state forward in time using the assigned process model
        """

        x_k = self.process_model.evaluate(x, u, dt)

        F_k = self.process_model.state_jacobian(x, u, dt)

        Q_k = self.process_model.covariance(x, u, dt)

        P_k = F_k @ p @ F_k.transpose(1, 2) + Q_k

        P_ks = .5 * (torch.add(P_k, P_k.transpose(1, 2)))

        return x_k, P_ks
    
    def correct(self, y: torch.Tensor, x_prior: list, p_prior: torch.Tensor, dt: float):
        """
        corrects the state with the incoming information
        """
        G_k = self.measurement_model.jacobian(x_prior)

        # TODO: need to formalize the noise() function in the measurement model
        R = self.measurement_model.noise(x_prior, y)
        s_k_inv = torch.linalg.inv(G_k @ p_prior @ G_k.transpose(1, 2) + R)
        K = p_prior @ G_k.transpose(1, 2) @ s_k_inv

        x_posterior, meas_res = self.measurement_model.update(x_prior, K, y)

        # calculate measurement-space covariance
        S = (G_k @ p_prior @ G_k.transpose(1, 2)) + R

        x2_val = (meas_res.transpose(1, 2) @ torch.linalg.inv(S) @ meas_res)

        if True: #(x2_val) < 3.841:
            p_posterior = (torch.eye(15).unsqueeze(0) - K @ G_k) @ p_prior
            P_ks = .5 * (torch.add(p_posterior, p_posterior.transpose(1, 2)))

            return x_posterior, P_ks, True
        else:
            print("X2 Test Failed, performing outlier rejection.")
            return x_prior, p_prior, False
# %%
