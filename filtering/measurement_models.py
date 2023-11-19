import torch
import torch.nn.functional as F
import numpy as np
import navlie
import scipy.constants
from pymlg.torch import SO3, SE23
from pymlg.numpy import SO3 as SO3_np
from pymlg.numpy import SE23 as SE23_np
import navlie as nav
import time

from network.resnet1d.model_resnet_tlio import ResNet1D, BasicBlock1D

from network.dido_preprocessor import generate_gravity_aligned_input

from network.modules import VelocityUnitVectorRegressor, VelocityVectorRegressor

from filtering import filtering_utils

from network.resnet1d.loss import gen_cov_with_activation_cholesky, gen_cov_tlio_form, gen_cov_diag_only

def fun(T: nav.lib.SE23State):
    # Returns the normalized body-frame velocity vector in gravity frame
    C_ab = T.attitude
    v_zw_a = T.velocity
    b_1_a_x = C_ab[0,0]
    b_1_a_y = C_ab[1,0]

    # You can check that SO3.wedge(g_1_a) @ b_1_a has zero z component
    # g_1_a = np.array([b_1_a_x, b_1_a_y, 0])
    # g_1_a = g_1_a / np.linalg.norm(g_1_a)
    # where b_1_a is the first column of C_ab

    #gamma = np.arccos(b_1_a_x/np.sqrt(b_1_a_x**2 + b_1_a_y**2))
    gamma = np.arctan2(b_1_a_y, b_1_a_x)

    C_ag = SO3_np.Exp([0,0,gamma])

    v_zw_g = C_ag.T @ v_zw_a
    v_zw_g_norm = v_zw_g / np.linalg.norm(v_zw_g)
    return v_zw_g_norm

def fun_vel(T: nav.lib.SE23State):
    # Returns the normalized body-frame velocity vector in gravity frame
    C_ab = T.attitude
    v_zw_a = T.velocity
    b_1_a_x = C_ab[0,0]
    b_1_a_y = C_ab[1,0]

    # You can check that SO3.wedge(g_1_a) @ b_1_a has zero z component
    # g_1_a = np.array([b_1_a_x, b_1_a_y, 0])
    # g_1_a = g_1_a / np.linalg.norm(g_1_a)
    # where b_1_a is the first column of C_ab

    #gamma = np.arccos(b_1_a_x/np.sqrt(b_1_a_x**2 + b_1_a_y**2))
    gamma = np.arctan2(b_1_a_y, b_1_a_x)

    C_ag = SO3_np.Exp([0,0,gamma])

    v_zw_g = C_ag.T @ v_zw_a
    return v_zw_g

def fun_1(gamma : np.ndarray):
    C_ag = SO3_np.Exp(np.array([0, 0, gamma]).reshape(3))

    return C_ag.T

def fun_2(C : nav.lib.SO3State):
    b_1_a_x = C.attitude[0, 0]
    b_1_a_y = C.attitude[1, 0]
    gamma = np.arctan2(b_1_a_y, b_1_a_x)

    return gamma

def fun_3(T: nav.lib.SE23State):
    # Returns the normalized body-frame velocity vector in gravity frame
    C_ab = T.attitude
    v_zw_a = T.velocity
    b_1_a_x = C_ab[0,0]
    b_1_a_y = C_ab[1,0]

    # You can check that SO3.wedge(g_1_a) @ b_1_a has zero z component
    # g_1_a = np.array([b_1_a_x, b_1_a_y, 0])
    # g_1_a = g_1_a / np.linalg.norm(g_1_a)
    # where b_1_a is the first column of C_ab

    #gamma = np.arccos(b_1_a_x/np.sqrt(b_1_a_x**2 + b_1_a_y**2))
    gamma = np.arctan2(b_1_a_y, b_1_a_x)

    C_ag = SO3_np.Exp([0,0,gamma])

    v_zw_g = C_ag.T @ v_zw_a
    return v_zw_g

def fun_4(T: nav.lib.SE23State):
    # Returns the normalized body-frame velocity vector in gravity frame
    C_ab = T.attitude
    v_zw_a = T.velocity

    return 1 / np.linalg.norm(v_zw_a)

def jac_right_charles(T: nav.lib.SE23State):
    """ For RIGHT perturbation. """
    C_ab = T.attitude
    v_zw_a = T.velocity
    T_ab = T.value
    b_1_a_x = C_ab[0,0]
    b_1_a_y = C_ab[1,0]
    gamma = np.arctan2(b_1_a_y, b_1_a_x)
    C_ag = SO3_np.Exp([0,0,gamma])
    v_zw_g = C_ag.T @ v_zw_a
    v_zw_g_norm = v_zw_g / np.linalg.norm(v_zw_g)

    D = np.block([np.eye(3), np.zeros((3,2))]) 
    e_4 = np.array([0,0,0,1,0]).reshape((-1,1))
    e_1 = np.array([1,0,0,0,0])
    e_3 = np.array([0,0,1])

    # jacobian of gamma wrt first column of C_ab
    dgammadb = -np.array([[-b_1_a_y/(b_1_a_x**2 + b_1_a_y**2), b_1_a_x/(b_1_a_x**2 + b_1_a_y**2), 0]])

    # jacobian of first column of C_ab wrt to xi
    dbdxi = D @ T_ab @ SE23_np.odot(e_1)

    # jacobian of v_zw_g wrt to xi
    J = SO3_np.left_jacobian([0,0,-gamma])
    dvgdxi = C_ag.T @ D @ T_ab @ SE23_np.odot(e_4) + C_ag.T @ SO3_np.cross(J @ e_3) @ D @ T_ab @ e_4 @ dgammadb @ dbdxi
    
    # jacobian of unit vector wrt v_zw_g (i.e. of x/norm(x))
    dvndvg = (np.eye(3) - np.outer(v_zw_g_norm, v_zw_g_norm))/ np.linalg.norm(v_zw_g)

    # jacobian of unit vector wrt xi
    dvndxi = dvndvg @ dvgdxi 
    return dvndxi

class RelativePosInGravityFrame:
    def __init__(self):
        self.relative_position_model = ResNet1D(
            BasicBlock1D, 6, 3, [2, 2, 2, 2], (400 // 32) + 1
        )

        self.params = torch.load(
            "/home/abajwa/mcgill/learned_quad_inertial/models/tlio_models/best_val_loss.pt"
        )

        self.relative_position_model.load_state_dict(
            self.params.get("model_state_dict")
        )
        self.relative_position_model.eval()

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is of shape (1, 15m), where m is the number of cloned states
        """
        # retrieve rotation and position vector from oldest clone state
        phi = x[:, 0:3]
        r_i = x[:, 6:9]

        # transform rotation vector into corresponding yaw-rotation matrix
        C_gamma = filtering_utils.c_3(phi)

        # retrieve current position
        r_j = x[:, -9:-6]

        r_ji = C_gamma.transpose(1, 2) @ (r_j - r_i).unsqueeze(2)

        return r_ji.reshape(3, 1)

    def jacobian(self, x: torch.Tensor, m: int) -> torch.Tensor:
        # zero-declare measurement jacobian
        G_k = torch.zeros(1, 3, 15 * (m + 1))

        # retrieve rotation and position vector from oldest clone state
        phi = x[:, 0:3]
        r_i = x[:, 6:9]

        # transform rotation vector into corresponding yaw-rotation matrix
        zero = phi.new_zeros(1)
        ones = phi.new_ones(1)
        C_gamma = filtering_utils.c_3(phi)

        # retrieve current position
        r_j = x[:, -3:]

        # jacobian wrt. r_i
        G_k[:, :, 6:9] = -C_gamma.transpose(1, 2)

        # jacobian wrt. r_j
        G_k[:, :, -9:-6] = C_gamma.transpose(1, 2)

        # jacobian wrt. C_i
        # transform rotation vector into corresponding yaw-rotation matrix
        C_z = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                torch.cos(phi[:, 2]) * torch.tan(phi[:, 1]),
                torch.sin(phi[:, 2]) * torch.tan(phi[:, 1]),
                ones,
            ),
            1,
        ).view(3, 3)

        # define auxiliary matrix
        G_k[:, :, 0:3] = C_gamma.transpose(1, 2) @ SO3.wedge(r_j - r_i) @ C_z

        return G_k

    def evaluate(
        self, y_inertial: torch.Tensor, x: torch.Tensor, C_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        y_inertial will be in the shape of [1, 6, N]. returns a tuple of ( (1, 3) (1, 3) ) torch.Tensor instances representing prediction
        and associated covariance
        """

        # rotate y_inertial into gravity-aligned frame of anchor state
        phi = x[:, 0:3]

        C_gamma = filtering_utils.c_3(phi)

        # generate rotation from body to local gravity-aligned frame
        C = C_gamma.transpose(1, 2) @ C_batch

        # rotate inertial window and reshape into individual channels
        omega = y_inertial[:, 0, :]
        acc = y_inertial[:, 1, :]

        omega_rot = torch.bmm(C, omega.unsqueeze(2))
        acc_rot = torch.bmm(C, acc.unsqueeze(2))

        i_ = torch.cat((omega_rot, acc_rot), dim=1).reshape(1, 6, -1)

        r_ij, w_r_ij = self.relative_position_model(i_)

        return r_ij.reshape(3, 1), torch.diagflat(torch.abs(w_r_ij * 10)).unsqueeze(0)

class VelocityVector:
    def __init__(
        self,
        args,
        Q_c: torch.Tensor,
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # the projection matrix for the velocity vector into x and y
        self.Q_c = Q_c

        # the gravity vector in the inertial frame
        self.g_a = g_a.reshape(1, 3, 1)

        self.perturbation = args.perturbation

        self.args = args

        self.meas = None
        self.meas_res = None

        # retrieve model
        self.model = VelocityVectorRegressor.load_from_checkpoint(
            checkpoint_path=args.model_path
        )
        self.model.eval()

    def measure(self, x: list) -> torch.Tensor:
        """
        Implementation of g(x). Returns the projected measurement given a state

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        g(x_{k}^) : torch.Tensor
            The projected measurement given the prior state
        """

        C_k, v_k, r_k = SE23.to_components(x[0])

        c21 = C_k[:, 1, 0]
        c11 = C_k[:, 0, 0]

        # # compute overall rotation for imu samples
        # anchor_euler = SO3.to_euler(C_k, order="123")
        # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
        gamma = torch.arctan2(c21, c11)
        phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

        C_ag = SO3.Exp(phi_gamma)

        v_zw_g = (C_ag.transpose(1, 2) @ v_k.reshape(3, 1))

        return v_zw_g.reshape(1, 3, 1)

    def jacobian(self, x: list) -> torch.Tensor:
        C_k, v_k, r_k = SE23.to_components(x[0])

        # generate analytical jacobian for comparison
        if self.perturbation == "right":
            G_vu = torch.zeros(1, 3, 15)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            # # compute overall rotation for imu samples
            # anchor_euler = SO3.to_euler(C_k, order="123")
            # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            gamma = torch.arctan2(c21, c11)
            phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

            C_ag = SO3.Exp(phi_gamma)
            alpha = 1 / torch.norm(v_k, dim=1, keepdim=True)

            v_zw_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)
            v_zw_g = C_ag.transpose(1, 2) @ v_k.reshape(3, 1)

            coeff_y = c11 / (c11**2 + c21**2)
            coeff_x = -c21 / (c11**2 + c21**2)

            B = torch.Tensor([1, 0, 0]).view(1, 3, 1)
            A = torch.Tensor([0, 1, 0]).view(1, 1, 3)
            J = torch.Tensor([1, 0, 0]).view(1, 1, 3)

            d_gamma_phi = -coeff_y * (A @ C_k @ SO3.wedge(B)) - coeff_x * (
                J @ C_k @ SO3.wedge(B)
            )

            d_gamma_phi_v = (v_k @ d_gamma_phi).view(1, 3, 3)

            d_C_gamma = (C_ag @ SO3.wedge(torch.Tensor([0, 0, 1]).view(1, 3, 1))).transpose(1, 2)

            G_vu[:, :, 0:3] = (d_C_gamma @ d_gamma_phi_v)

            G_vu[:, :, 3:6] = (C_ag.transpose(1, 2) @ C_k)

        else:
            G_vu = torch.zeros(1, 3, 15)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            # # compute overall rotation for imu samples
            # anchor_euler = SO3.to_euler(C_k, order="123")
            # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            gamma = torch.arctan2(c21, c11)
            phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

            C_ag = SO3.Exp(phi_gamma)
            alpha = 1 / torch.norm(v_k, dim=1, keepdim=True)

            v_zw_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)
            v_zw_g = C_ag.transpose(1, 2) @ v_k.reshape(3, 1)

            coeff_y = c11 / (c11**2 + c21**2)
            coeff_x = -c21 / (c11**2 + c21**2)

            B = torch.Tensor([1, 0, 0]).view(1, 3, 1)
            A = torch.Tensor([0, 1, 0]).view(1, 1, 3)
            J = torch.Tensor([1, 0, 0]).view(1, 1, 3)

            d_gamma_phi = -coeff_y * (A @ SO3.wedge(C_k @ B)) - coeff_x * (
                J @ SO3.wedge(C_k @ B)
            )

            d_C_gamma = (C_ag @ SO3.wedge(torch.Tensor([0, 0, 1]).view(1, 3, 1))).transpose(1, 2)

            G_vu[:, :, 0:3] = -C_ag.transpose(1, 2) @ SO3.wedge(v_k) + d_C_gamma @ v_k @ d_gamma_phi

            G_vu[:, :, 3:6] = C_ag.transpose(1, 2)

        # # based on perturbation direction, compute numeric jacobian
        if (self.args.check_numeric):
            if self.perturbation == "right":
                T = nav.lib.SE23State(SE23.Log(x[0]).numpy().reshape(9, 1), direction="right")
            else:
                T = nav.lib.SE23State(SE23.Log(x[0]).numpy().reshape(9, 1), direction="left")

            # # for now, just using numeric jacobian
            jac_fd = nav.jacobian(fun_vel, T)
            jac_fd = torch.Tensor(jac_fd).unsqueeze(0)

            print(torch.norm(jac_fd[:, :, :9] - G_vu[:, :, :9]))

        return G_vu

    def generate_model_input(
        self, y: torch.Tensor, x_prior: list
    ) -> torch.Tensor:
        """
        from current state estimate and IMU sample history, generate input for the velocity unit vector regressor
        """

        C_0, _, _ = SE23.to_components(x_prior[0])

        # y is incoming IMU sample tensor with shape (6, N), need to split into gyroscope and accelerometer samples
        gyro = y[0:3, :]
        acc = y[3:6, :]
        ts = y[6, :]

        return generate_gravity_aligned_input(gyro=gyro, acc=acc, ts=ts, C_0=C_0)

    def update(
        self, x_prior: list, K: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        State update function for the measurement model.
        """

        with torch.no_grad():

            self.meas_res = self.meas.unsqueeze(2) - self.measure(x_prior)

            self.innov = K @ self.meas_res

            # seperate ksi from bias update
            ksi = self.innov[:, :9, :]
            bias_update = self.innov[:, 9:, :]

            # update state
            if self.perturbation == "right":
                lie_posterior = x_prior[0] @ SE23.Exp(ksi)
            else:
                lie_posterior = SE23.Exp(ksi) @ x_prior[0]
            bias_posterior = x_prior[1] + bias_update

            x_posterior = list((lie_posterior, bias_posterior))

            return x_posterior, self.meas_res

    def noise(self, x_prior: list, y: torch.Tensor):
        with torch.no_grad():

            # preprocess input data and generate model input
            model_y = self.generate_model_input(y, x_prior).unsqueeze(0)

            self.meas, cov_v_g = self.model.net(model_y.to(self.args.device))

            self.meas = self.meas.cpu()

            # convert to cpu, this results in a (1, 6) tensor that must be activated as done in training
            R = self.args.cov_scaling * gen_cov_diag_only(cov_v_g, self.args.device)

            R = R.view(1, 3, 3).cpu()

            # print(R)

            return R

class VelocityUnitVector:
    def __init__(
        self,
        args,
        Q_c: torch.Tensor,
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # the projection matrix for the velocity vector into x and y
        self.Q_c = Q_c

        # the gravity vector in the inertial frame
        self.g_a = g_a.reshape(1, 3, 1)

        self.perturbation = args.perturbation

        self.args = args

        self.meas = None
        self.meas_res = None

        # retrieve model
        self.model = VelocityUnitVectorRegressor.load_from_checkpoint(
            checkpoint_path=args.model_path
        )
        self.model.eval()

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of g(x). Returns the projected measurement given a state

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        g(x_{k}^) : torch.Tensor
            The projected measurement given the prior state
        """

        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        c21 = C_k[:, 1, 0]
        c11 = C_k[:, 0, 0]

        # # compute overall rotation for imu samples
        # anchor_euler = SO3.to_euler(C_k, order="123")
        # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
        gamma = torch.arctan2(c21, c11)
        phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

        C_ag = SO3.Exp(phi_gamma)
        alpha = torch.norm(v_k, dim=1, keepdim=True)

        v_zw_g = (C_ag.transpose(1, 2) @ v_k.reshape(3, 1)) / alpha

        return v_zw_g.reshape(1, 3, 1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        # generate analytical jacobian for comparison
        if self.perturbation == "right":
            G_vu = torch.zeros(1, 3, 15)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            # # compute overall rotation for imu samples
            # anchor_euler = SO3.to_euler(C_k, order="123")
            # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            gamma = torch.arctan2(c21, c11)
            phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

            C_ag = SO3.Exp(phi_gamma)
            alpha = 1 / torch.norm(v_k, dim=1, keepdim=True)

            v_zw_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)
            v_zw_g = C_ag.transpose(1, 2) @ v_k.reshape(3, 1)

            coeff_y = c11 / (c11**2 + c21**2)
            coeff_x = -c21 / (c11**2 + c21**2)

            B = torch.Tensor([1, 0, 0]).view(1, 3, 1)
            A = torch.Tensor([0, 1, 0]).view(1, 1, 3)
            J = torch.Tensor([1, 0, 0]).view(1, 1, 3)

            d_gamma_phi = -coeff_y * (A @ C_k @ SO3.wedge(B)) - coeff_x * (
                J @ C_k @ SO3.wedge(B)
            )

            d_gamma_phi_v = (v_k @ d_gamma_phi).view(1, 3, 3)

            d_C_gamma = (C_ag @ SO3.wedge(torch.Tensor([0, 0, 1]).view(1, 3, 1))).transpose(1, 2)

            G_vu[:, :, 0:3] = (d_C_gamma @ d_gamma_phi_v) * alpha

            g_vu_02 = (v_zw_g @ v_zw_b.transpose(1, 2)) * alpha**3
            G_vu[:, :, 3:6] = ((C_ag.transpose(1, 2) @ C_k) * alpha) - g_vu_02

        else:
            G_vu = torch.zeros(1, 3, 15)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            # # compute overall rotation for imu samples
            # anchor_euler = SO3.to_euler(C_k, order="123")
            # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            gamma = torch.arctan2(c21, c11)
            phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

            C_ag = SO3.Exp(phi_gamma)
            alpha = 1 / torch.norm(v_k, dim=1, keepdim=True)

            v_zw_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)
            v_zw_g = C_ag.transpose(1, 2) @ v_k.reshape(3, 1)

            coeff_y = c11 / (c11**2 + c21**2)
            coeff_x = -c21 / (c11**2 + c21**2)

            B = torch.Tensor([1, 0, 0]).view(1, 3, 1)
            A = torch.Tensor([0, 1, 0]).view(1, 1, 3)
            J = torch.Tensor([1, 0, 0]).view(1, 1, 3)

            d_gamma_phi = -coeff_y * (A @ SO3.wedge(C_k @ B)) - coeff_x * (
                J @ SO3.wedge(C_k @ B)
            )

            d_C_gamma = (C_ag @ SO3.wedge(torch.Tensor([0, 0, 1]).view(1, 3, 1))).transpose(1, 2)

            G_vu[:, :, 0:3] = -C_ag.transpose(1, 2) @ SO3.wedge(v_k) * alpha + d_C_gamma @ v_k @ d_gamma_phi * alpha

            G_vu[:, :, 3:6] = C_ag.transpose(1, 2) * alpha - C_ag.transpose(1, 2) @ v_k @ v_k.transpose(1, 2) * alpha**3

        # # based on perturbation direction, compute numeric jacobian
        if (self.args.check_numeric):
            if self.perturbation == "right":
                T = nav.lib.SE23State(x[:, :9, :].numpy().reshape(9, 1), direction="right")
            else:
                T = nav.lib.SE23State(x[:, :9, :].numpy().reshape(9, 1), direction="left")

            # # for now, just using numeric jacobian
            jac_fd = nav.jacobian(fun, T)
            jac_fd = torch.Tensor(jac_fd).unsqueeze(0)

            print(torch.norm(jac_fd[:, :, :9] - G_vu[:, :, :9]))

        return G_vu

    def generate_model_input(
        self, y: torch.Tensor, x_prior: torch.Tensor
    ) -> torch.Tensor:
        """
        from current state estimate and IMU sample history, generate input for the velocity unit vector regressor
        """

        T_0 = SE23.Exp(x_prior[:, :9, :])

        C_0, _, _ = SE23.to_components(T_0)

        # y is incoming IMU sample tensor with shape (6, N), need to split into gyroscope and accelerometer samples
        gyro = y[0:3, :]
        acc = y[3:6, :]
        ts = y[6, :]

        return generate_gravity_aligned_input(gyro=gyro, acc=acc, ts=ts, C_0=C_0)

    def update(
        self, x_prior: torch.Tensor, K: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        State update function for the measurement model.
        """

        with torch.no_grad():

            self.meas = (self.meas / torch.norm(self.meas, dim=1).unsqueeze(1)).cpu()

            self.meas_res = self.meas.unsqueeze(2) - self.measure(x_prior)

            self.innov = K @ self.meas_res

            # seperate ksi from bias update
            ksi = self.innov[:, :9, :]
            bias_update = self.innov[:, 9:, :]

            # update state
            if self.perturbation == "right":
                lie_posterior = SE23.Exp(x_prior[:, :9, :]) @ SE23.Exp(ksi)
            else:
                lie_posterior = SE23.Exp(ksi) @ SE23.Exp(x_prior[:, :9, :])
            bias_posterior = x_prior[:, 9:, :] + bias_update

            x_posterior = torch.cat((SE23.Log(lie_posterior), bias_posterior), dim=1)

            return x_posterior, self.meas_res

    def noise(self, x_prior: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():

            # preprocess input data and generate model input
            model_y = self.generate_model_input(y, x_prior).unsqueeze(0)

            self.meas, cov_v_g = self.model.net(model_y.to(self.args.device))

            # convert to cpu, this results in a (1, 6) tensor that must be activated as done in training
            R = self.args.cov_scaling * gen_cov_tlio_form(cov_v_g, self.args.device)

            R = R.view(1, 3, 3).cpu()

            return R

class SyntheticVelocityUnitVector:
    def __init__(
        self,
        args,
        Q_c: torch.Tensor,
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # the projection matrix for the velocity vector into x and y
        self.Q_c = Q_c

        # the gravity vector in the inertial frame
        self.g_a = g_a.reshape(1, 3, 1)

        # marker for classification state at each timestep
        self.marker = torch.zeros(5)

        self.perturbation = args.perturbation

        self.args = args

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of g(x). Returns the projected measurement given a state

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        g(x_{k}^) : torch.Tensor
            The projected measurement given the prior state
        """

        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        # generate velocity unit vector in the body-frame
        v_k_u = (C_k.transpose(1, 2) @ v_k.reshape(3, 1)) / torch.norm(
            v_k, dim=1, keepdim=True
        )

        return v_k_u.reshape(1, 3, 1)

    def measure_np(self, x: np.ndarray) -> np.ndarray:
        """
        a np clone of the torch measurement function for numeric jacobian testing with navlie
        """

        x = torch.Tensor(x)

        g_x = self.measure(x)

        return g_x.numpy()

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        # generate analytical jacobian for comparison
        if self.perturbation == "right":
            G_vu = torch.zeros(1, 3, 15)

            alpha = torch.norm(v_k, dim=1, keepdim=True)
            v_k_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

            G_vu[:, :, 0:3] = SO3.wedge(v_k_b) / alpha

            g_vu_02 = v_k_b @ v_k_b.transpose(1, 2) / alpha**3
            G_vu[:, :, 3:6] = (torch.eye(3).unsqueeze(0) / alpha) - g_vu_02
        else:
            G_vu = torch.zeros(1, 3, 15)

            alpha = torch.norm(v_k, dim=1, keepdim=True)
            v_k_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

            g_vu_02 = v_k_b @ v_k.transpose(1, 2) / alpha**3
            G_vu[:, :, 3:6] = (C_k.transpose(1, 2) / alpha) - g_vu_02

        # for now, just using numeric jacobian
        # G_k_numeric = navlie.utils.jacobian(self.measure_np, x.detach().numpy())

        # G_k_numeric_batch = torch.Tensor(G_k_numeric).unsqueeze(0)

        # # compare analytical and numeric jacobian
        # print(torch.norm(G_k_numeric_batch - G_vu))
        # print("analytical g_vu_v", G_vu[:, :, 0:3])
        # print("numeric g_vu_v", G_k_numeric_batch[:, :, 0:3])

        # testing with charles' jac
        # D = np.block([np.eye(3), np.zeros((3,2))])
        # b = np.array([0,0,0,1,0]).reshape((-1,1))
        # T_inv = SE23.inverse(SE23.Exp(x[:, :9, :]))
        # T_inv = T_inv.squeeze(0).detach().numpy()
        # z = - D @ T_inv @ b
        # jac_true = (np.eye(3)/(np.linalg.norm(z)) - (z @ z.T)/(np.linalg.norm(z)**3)) @ D @ pymlg.numpy.SE23.odot(T_inv @ b)

        # print(torch.norm(torch.Tensor(jac_true) - G_vu[:, :, :9]))

        return G_vu

    def update(
        self, x_prior: torch.Tensor, K: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        State update function for the measurement model.
        """
        meas_res = y - self.measure(x_prior)
        self.innov = K @ meas_res

        # seperate ksi from bias update
        ksi = self.innov[:, :9, :]
        bias_update = self.innov[:, 9:, :]

        # update state
        if self.perturbation == "right":
            lie_posterior = SE23.Exp(x_prior[:, :9, :]) @ SE23.Exp(ksi)
        else:
            lie_posterior = SE23.Exp(ksi) @ SE23.Exp(x_prior[:, :9, :])
        bias_posterior = x_prior[:, 9:, :] + bias_update

        x_posterior = torch.cat((SE23.Log(lie_posterior), bias_posterior), dim=1)

        return x_posterior, meas_res

    def noise(self, x_prior: torch.Tensor, y: torch.Tensor):
        R = torch.eye(3).unsqueeze(0) * self.args.sigma_v_u**2

        return R


class SyntheticVelocityUnitVectorGravityAligned:
    def __init__(
        self,
        args,
        Q_c: torch.Tensor,
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # the projection matrix for the velocity vector into x and y
        self.Q_c = Q_c

        # the gravity vector in the inertial frame
        self.g_a = g_a.reshape(1, 3, 1)

        # marker for classification state at each timestep
        self.marker = torch.zeros(5)

        self.perturbation = args.perturbation

        self.args = args

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of g(x). Returns the projected measurement given a state

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        g(x_{k}^) : torch.Tensor
            The projected measurement given the prior state
        """

        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        c21 = C_k[:, 1, 0]
        c11 = C_k[:, 0, 0]

        # # compute overall rotation for imu samples
        # anchor_euler = SO3.to_euler(C_k, order="123")
        # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
        gamma = torch.arctan2(c21, c11)
        phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

        C_ag = SO3.Exp(phi_gamma)
        alpha = torch.norm(v_k, dim=1, keepdim=True)

        v_zw_g = (C_ag.transpose(1, 2) @ v_k.reshape(3, 1)) / alpha

        return v_zw_g.reshape(1, 3, 1)

    def measure_np(self, x: np.ndarray) -> np.ndarray:
        """
        a np clone of the torch measurement function for numeric jacobian testing with navlie
        """

        x = torch.Tensor(x)

        anchor_euler = SO3.to_euler(SO3.Exp(x), order="123")

        return anchor_euler.numpy()

    def numeric_gamma_mapping(self, x: np.ndarray) -> np.ndarray:
        x = torch.Tensor(x)

        anchor_euler = SO3.to_euler(SO3.Exp(x), order="123")

        return anchor_euler.numpy()

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        # generate analytical jacobian for comparison
        if self.perturbation == "right":
            G_vu = torch.zeros(1, 3, 15)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            # # compute overall rotation for imu samples
            # anchor_euler = SO3.to_euler(C_k, order="123")
            # C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            gamma = torch.arctan2(c21, c11)
            phi_gamma = torch.Tensor([0, 0, gamma]).view(1, 3, 1)

            C_ag = SO3.Exp(phi_gamma)
            alpha = 1 / torch.norm(v_k, dim=1, keepdim=True)

            v_zw_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)
            v_zw_g = C_ag.transpose(1, 2) @ v_k.reshape(3, 1)

            c21 = C_k[:, 1, 0]
            c11 = C_k[:, 0, 0]

            coeff_y = c11 / (c11**2 + c21**2)
            coeff_x = -c21 / (c11**2 + c21**2)

            B = torch.Tensor([1, 0, 0]).view(1, 3, 1)
            A = torch.Tensor([0, 1, 0]).view(1, 1, 3)
            J = torch.Tensor([1, 0, 0]).view(1, 1, 3)

            d_gamma_phi = -coeff_y * (A @ C_k @ SO3.wedge(B)) - coeff_x * (
                J @ C_k @ SO3.wedge(B)
            )

            # trying derivative temporarily - from https://arxiv.org/pdf/1312.0788.pdf
            d_gamma_phi_v = (v_k @ d_gamma_phi).view(1, 3, 3)

            d_C_gamma = (C_ag @ SO3.wedge(torch.Tensor([0, 0, 1]).view(1, 3, 1))).transpose(1, 2)

            G_vu[:, :, 0:3] = (d_C_gamma @ d_gamma_phi_v) * alpha

            g_vu_02 = (v_zw_g @ v_zw_b.transpose(1, 2)) * alpha**3
            G_vu[:, :, 3:6] = ((C_ag.transpose(1, 2) @ C_k) * alpha) - g_vu_02

        else:
            G_vu = torch.zeros(1, 3, 15)
            # compute overall rotation for imu samples
            anchor_euler = SO3.to_euler(C_k, order="123")
            C_gamma = filtering_utils.c_3(anchor_euler.unsqueeze(2))
            C_beta = filtering_utils.c_2(anchor_euler.unsqueeze(2))
            C_alpha = filtering_utils.c_1(anchor_euler.unsqueeze(2))

            alpha = torch.norm(v_k, dim=1, keepdim=True)
            v_k_g = C_gamma.transpose(1, 2) @ v_k.reshape(3, 1)

            g_vu_02 = v_k_g @ v_k.transpose(1, 2) / alpha**3
            G_vu[:, :, 3:6] = (
                (C_beta @ C_alpha @ C_k.transpose(1, 2)) / alpha
            ) - g_vu_02

        # # based on perturbation direction, compute numeric jacobian
        if self.perturbation == "right":
            T = nav.lib.SE23State(x[:, :9, :].numpy().reshape(9, 1), direction="right")
        else:
            T = nav.lib.SE23State(x[:, :9, :].numpy().reshape(9, 1), direction="left")

        # for now, just using numeric jacobian
        # jac_fd = nav.jacobian(fun, T)
        # jac_fd = torch.Tensor(jac_fd).unsqueeze(0)
        jac_right = jac_right_charles(T)
        jac_right = torch.Tensor(jac_right).unsqueeze(0)

        # jac_C_ga_v = nav.jacobian(fun_3, T)
        # jac_C_ga_v = torch.Tensor(jac_C_ga_v).unsqueeze(0)

        # jac_norm = nav.jacobian(fun_4, T)
        # jac_norm = torch.Tensor(jac_norm).unsqueeze(0)
        # print(jac_norm[:, :, 3:6])
        # print(norm_jacob_test)
        # print(torch.norm(jac_norm[:, :, 3:6] - norm_jacob_test))

        # # compare analytical and numeric jacobian
        print(torch.norm(jac_right[:, :, :9] - G_vu[:, :, :9]))

        return G_vu

    def update(
        self, x_prior: torch.Tensor, K: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        State update function for the measurement model.
        """
        meas_res = y - self.measure(x_prior)
        self.innov = K @ meas_res

        # perform NIS outlier rejection

        # seperate ksi from bias update
        ksi = self.innov[:, :9, :]
        bias_update = self.innov[:, 9:, :]

        # update state
        if self.perturbation == "right":
            lie_posterior = SE23.Exp(x_prior[:, :9, :]) @ SE23.Exp(ksi)
        else:
            lie_posterior = SE23.Exp(ksi) @ SE23.Exp(x_prior[:, :9, :])
        bias_posterior = x_prior[:, 9:, :] + bias_update

        x_posterior = torch.cat((SE23.Log(lie_posterior), bias_posterior), dim=1)

        return x_posterior, meas_res

    def noise(self, x_prior: torch.Tensor, y: torch.Tensor):
        R = torch.eye(3).unsqueeze(0) * self.args.sigma_v_u**2

        return R


class NullQuadrotorMeasurements:
    def __init__(
        self,
        args,
        Q_c: torch.Tensor,
        g_a=torch.tensor([[0], [0], [-scipy.constants.g]]),
    ):
        # the projection matrix for the velocity vector into x and y
        self.Q_c = Q_c

        # the gravity vector in the inertial frame
        self.g_a = g_a.reshape(1, 3, 1)

        # marker for classification state at each timestep
        self.marker = torch.zeros(5)

        self.perturbation = args.perturbation

        self.args = args

    def assign_u(self, omega: torch.Tensor, a: torch.Tensor):
        self.omega = omega
        self.a_b = a

    def measure_np(self, x: np.ndarray) -> np.ndarray:
        """
        a np clone of the torch measurement function for numeric jacobian testing with navlie
        """

        x = torch.Tensor(x)

        g_x = self.measure(x)

        return g_x.numpy()

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of g(x). Returns the projected measurement given a state

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        g(x_{k}^) : torch.Tensor
            The projected measurement given the prior state
        """

        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        g_x = torch.empty(1, 0, 1)

        # if angular velocity is null, then add current debiased angular velocity measurement to vector
        if self.marker[0]:
            omega_bar = self.omega - x[:, 9:12, :]
            g_x = torch.cat((g_x, omega_bar.view(1, 3, 1)), dim=1)

        # if zero-velocity update is occuring, then add inertial linear acceleration and velocity measurements to vector
        if self.marker[1]:
            r0 = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

            a_b_bar = self.a_b - x[:, 12:15, :]

            r1 = a_b_bar + C_k.transpose(1, 2) @ self.g_a

            z_v_m = torch.cat((r0, r1), dim=1)

            g_x = torch.cat((g_x, z_v_m), dim=1)

        else:
            # if planar motion update is occuring, add upwards velocity estimate to measurement vector
            if self.marker[2]:
                # if body-frame planar motion update is occuring, then compute v_k_b and add to measurement vector
                # otherwise, add upwards inertial velocity estimate to measurement vector
                v_k_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

                g_x = torch.cat((g_x, v_k_b[:, 2, :].view(1, 1, 1)), dim=1)

            # if lateral motion update is occuring, add lateral velocity estimate to measurement vector
            if self.marker[3]:
                v_k_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

                g_x = torch.cat((g_x, v_k_b[:, 1, :].view(1, 1, 1)), dim=1)

            # if forward motion update is occuring, add forward velocity estimate to measurement vector
            if self.marker[4]:
                v_k_b = C_k.transpose(1, 2) @ v_k.reshape(3, 1)

                g_x = torch.cat((g_x, v_k_b[:, 0, :].view(1, 1, 1)), dim=1)

        return g_x

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns the measurement jacobian at time k, given the prior.

        Parameters
        ----------
        x : torch.Tensor
            The previous state. Expected to be an SE_2(3) x R^6 object with shape [N, 15, 1]

        Returns
        -------
        G_k : torch.Tensor
            The measurement jacobian at time k, a tensor with shape [N, 2, 15]
        """

        G_k = torch.empty(1, 0, 15)

        C_k, v_k, r_k = SE23.to_components(SE23.Exp(x[:, :9, :]))

        # if angular velocity is null, then add (1, 3, 15) measurement jacobian to overall jacobian
        if self.marker[0]:
            z_angVel_j = torch.zeros(1, 3, 15)

            z_angVel_j[:, :, 9:12] = -torch.eye(3, 3).unsqueeze(0)

            G_k = torch.cat((G_k, z_angVel_j), dim=1)

        # if zero-velocity update is occuring, then add (1, 6, 15) measurement jacobian to overall jacobian
        if self.marker[1]:
            r0 = torch.zeros((1, 3, 15))
            r1 = torch.zeros((1, 3, 15))

            if self.perturbation == "right":
                # wrt zero body-frame velocity
                r0[:, :, 0:3] = SO3.wedge(C_k.transpose(1, 2) @ v_k.reshape(3, 1))
                r0[:, :, 3:6] = torch.eye(3, 3).unsqueeze(0)

                # wrt zero body-frame acceleration (proper)
                r1[:, :, 0:3] = SO3.wedge(C_k.transpose(1, 2) @ self.g_a.reshape(3, 1))
                r1[:, :, 12:15] = -torch.eye(3, 3).unsqueeze(0)
            else:
                # wrt zero body-frame velocity
                r0[:, :, 3:6] = C_k.transpose(1, 2)

                # wrt zero body-frame acceleration (proper)
                r1[:, :, 0:3] = C_k.transpose(1, 2) @ SO3.wedge(self.g_a)
                r1[:, :, 12:15] = -torch.eye(3, 3).unsqueeze(0)

            z_vel_j = torch.cat((r0, r1), dim=1)

            G_k = torch.cat((G_k, z_vel_j), dim=1)

        else:
            # if planar motion update is occuring, add (1, 1, 15) measurement jacobian to overall jacobian
            if self.marker[2]:
                D = torch.tensor([[0.0, 0.0, 1.0]]).unsqueeze(0)

                z_vel_planar_j = torch.zeros(1, 1, 15)

                if self.perturbation == "right":
                    # jacobians for body-frame upwards velocity constraint
                    # delta_phi_k
                    z_vel_planar_j[:, :, 0:3] = D @ SO3.wedge(
                        C_k.transpose(1, 2) @ v_k.reshape(3, 1)
                    )

                    # delta_v_k
                    z_vel_planar_j[:, :, 3:6] = D @ torch.eye(3, 3).unsqueeze(0)

                else:
                    # jacobians for body-frame upwards velocity constraint
                    # delta_v_k
                    z_vel_planar_j[:, :, 3:6] = D @ C_k.transpose(1, 2)

                G_k = torch.cat((G_k, z_vel_planar_j), dim=1)

            # if lateral motion update is occuring, add (1, 1, 15) measurement jacobian to overall jacobian
            if self.marker[3]:
                D = torch.tensor([[0.0, 1.0, 0.0]]).unsqueeze(0)

                z_vel_lateral_j = torch.zeros(1, 1, 15)

                if self.perturbation == "right":
                    # delta_phi_k
                    z_vel_lateral_j[:, :, 0:3] = D @ SO3.wedge(
                        C_k.transpose(1, 2) @ v_k.reshape(3, 1)
                    )

                    # delta_v_k
                    z_vel_lateral_j[:, :, 3:6] = D @ torch.eye(3, 3).unsqueeze(0)

                else:
                    # delta_v_k
                    z_vel_lateral_j[:, :, 3:6] = D @ C_k.transpose(1, 2)

                G_k = torch.cat((G_k, z_vel_lateral_j), dim=1)

            if self.marker[4]:
                D = torch.tensor([[1.0, 0.0, 0.0]]).unsqueeze(0)

                z_vel_forward_j = torch.zeros(1, 1, 15)

                if self.perturbation == "right":
                    # delta_phi_k
                    z_vel_forward_j[:, :, 0:3] = D @ SO3.wedge(
                        C_k.transpose(1, 2) @ v_k.reshape(3, 1)
                    )

                    # delta_v_k
                    z_vel_forward_j[:, :, 3:6] = D @ torch.eye(3, 3).unsqueeze(0)

                else:
                    # delta_v_k
                    z_vel_forward_j[:, :, 3:6] = D @ C_k.transpose(1, 2)

                G_k = torch.cat((G_k, z_vel_forward_j), dim=1)

        # compute numeric jacobian for same measurement instance and print squared diff.
        # G_k_numeric = navlie.utils.jacobian(self.measure_np, x.detach().numpy())

        # G_k_numeric_batch = torch.Tensor(G_k_numeric).unsqueeze(0)

        # print(self.marker)
        # print("G_k", G_k)
        # print("G_k_numeric", G_k_numeric_batch)
        # print(torch.norm(G_k - G_k_numeric_batch))
        # print(G_k - G_k_numeric_batch)

        return G_k

    def update(
        self, x_prior: torch.Tensor, K: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        State update function for the measurement model.
        """
        meas_res = y - self.measure(x_prior)
        self.innov = K @ meas_res

        # seperate ksi from bias update
        ksi = self.innov[:, :9, :]
        bias_update = self.innov[:, 9:, :]

        # update state
        if self.perturbation == "right":
            lie_posterior = SE23.Exp(x_prior[:, :9, :]) @ SE23.Exp(ksi)
        else:
            lie_posterior = SE23.Exp(ksi) @ SE23.Exp(x_prior[:, :9, :])
        bias_posterior = x_prior[:, 9:, :] + bias_update

        x_posterior = torch.cat((SE23.Log(lie_posterior), bias_posterior), dim=1)

        return x_posterior, meas_res

    def noise(self, dt: float):
        Q_d = self.Q_c / dt

        g_x = torch.empty(1, 0, 0)
        d = 0

        # if angular velocity is null, then add noise corresponding to gyroscope white noise
        if self.marker[0]:
            Q_ang_vel = torch.eye(3, 3).unsqueeze(0)

            Q_ang_vel[:, 0:3, 0:3] *= self.args.sigma_omega

            g_x = F.pad(g_x, (0, 3, 0, 3), mode="constant", value=0)
            g_x[:, d : d + 3, d : d + 3] = Q_ang_vel
            d += 3

        # if zero-velocity update is occuring, then add noise corresponding to epsilon and accelerometer white noise
        if self.marker[1]:
            Q_z_vel = torch.eye(6, 6).unsqueeze(0)

            Q_z_vel[:, 0:3, 0:3] *= self.args.sigma_z_vel_null

            Q_z_vel[:, 3:6, 3:6] *= self.args.sigma_z_vel_rp

            g_x = F.pad(g_x, (0, 6, 0, 6), mode="constant", value=0)
            g_x[:, d : d + 6, d : d + 6] = Q_z_vel
            d += 6

        else:
            # if planar motion update is occuring, add upwards velocity estimate to measurement vector
            if self.marker[2]:
                Q_z_vel_planar = torch.ones((1, 1, 1)) * self.args.sigma_z_vel_planar

                g_x = F.pad(g_x, (0, 1, 0, 1), mode="constant", value=0)
                g_x[:, d : d + 1, d : d + 1] = Q_z_vel_planar
                d += 1

            # if lateral motion update is occuring, add lateral velocity estimate to measurement vector
            # TODO: currently the same noise added as the planar velocity, but this should be changed at some point
            if self.marker[3]:
                Q_z_vel_lateral = torch.ones((1, 1, 1)) * self.args.sigma_z_vel_lateral

                g_x = F.pad(g_x, (0, 1, 0, 1), mode="constant", value=0)
                g_x[:, d : d + 1, d : d + 1] = Q_z_vel_lateral
                d += 1

            # if forward motion update is occuring, add forward velocity estimate noise to R
            if self.marker[4]:
                Q_z_vel_upwards = torch.ones((1, 1, 1)) * self.args.sigma_z_vel_lateral

                g_x = F.pad(g_x, (0, 1, 0, 1), mode="constant", value=0)
                g_x[:, d : d + 1, d : d + 1] = Q_z_vel_upwards
                d += 1

        return g_x
