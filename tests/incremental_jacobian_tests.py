# %%
from navlie.lib import (
    BodyVelocityIncrement,
    IMUIncrement,
    PreintegratedBodyVelocity,
    PreintegratedIMUKinematics,
    LinearIncrement,
    PreintegratedLinearModel,
    IMU,
    IMUKinematics,
    IMUState,
    BodyFrameVelocity,
    DoubleIntegrator,
    VectorInput,
    SE3State,
    VectorState,
)
from navlie.lib.models import DoubleIntegratorWithBias
from navlie.filters import ExtendedKalmanFilter
import numpy as np
from pymlg import SE23 as SE23np
from pymlg import SO3 as SO3np
from pymlg.torch import SE23, SO3
from navlie import StateWithCovariance, ExtendedKalmanFilter

from navlie.lib.imu import L_matrix, adjoint_IE3, inverse_IE3, U_matrix, U_matrix_inv, G_matrix, G_matrix_inv, N_matrix

import torch
from filtering.process_models import CoupledIMUKinematicModel
from filtering.filtering_utils import form_N_matrix

def test_preintegration_functions():
    for i in range(100):
        gyro = torch.rand(3)
        acc = torch.rand(3)

        N = N_matrix(gyro.numpy())

        u = torch.cat((gyro.unsqueeze(0), acc.unsqueeze(0)), dim=0).unsqueeze(0)
        N_torch = form_N_matrix(torch.Tensor(gyro).reshape(1, 3, 1))

        assert torch.allclose(torch.Tensor(N), N_torch, atol=1e-6), "N matrix failed"

    for i in range(100):
        gyro = torch.rand(3)
        acc = torch.rand(3)

        L = L_matrix(gyro.numpy(), acc.numpy(), .1)

        u = torch.cat((gyro.unsqueeze(0), acc.unsqueeze(0)), dim=0).unsqueeze(0)
        L_torch = CoupledIMUKinematicModel.input_jacobian_pose(u, .1)

        assert torch.allclose(torch.Tensor(L), L_torch, atol=1e-6), "L matrix failed"

    for i in range(100):
        gyro = torch.rand(3)
        acc = torch.rand(3)

        U = U_matrix(gyro.numpy(), acc.numpy(), .1)

        u = torch.cat((gyro.unsqueeze(0), acc.unsqueeze(0)), dim=0).unsqueeze(0)
        U_torch = CoupledIMUKinematicModel.generate_u(u, .1)

        assert torch.allclose(torch.Tensor(U), U_torch, atol=1e-6), "U matrix failed"

    for i in range(100):
        gyro = torch.rand(3)
        acc = torch.rand(3)

        U = U_matrix(gyro.numpy(), acc.numpy(), .1)
        U_inv = inverse_IE3(U)
        A = adjoint_IE3(U_inv)

        u = torch.cat((gyro.unsqueeze(0), acc.unsqueeze(0)), dim=0).unsqueeze(0)
        U_torch = CoupledIMUKinematicModel.generate_u(u, .1)
        U_inv_torch = CoupledIMUKinematicModel.ie3_inv(U_torch)
        A_torch = CoupledIMUKinematicModel.ie3_adj(U_inv_torch)

        assert torch.allclose(torch.Tensor(U_inv), U_inv_torch, atol=1e-6), "IE3 inverse failed"
        assert torch.allclose(torch.Tensor(A), A_torch, atol=1e-6), "IE3 adjoint failed"

def test_imu_preintegration_equivalence(direction):
    """
    Tests to make sure IMU preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(12)
    accel_bias = [0, 0, 0]
    gyro_bias = [0, 0, 0]

    model = IMUKinematics(Q)
    dt = 1 / 400
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23np.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        gyro_bias,
        accel_bias,
        0,
        direction=direction,
    )
    P0 = np.identity(15)
    ekf = ExtendedKalmanFilter(model)
    x0 = StateWithCovariance(x, P0)
    rmi = IMUIncrement(Q / dt, gyro_bias=gyro_bias, accel_bias=accel_bias)
    preint_model = PreintegratedIMUKinematics()

    # DIVE preintegration model setup
    dive_preint_model = CoupledIMUKinematicModel(torch.Tensor(Q).unsqueeze(0), perturbation=direction)
    dive_preint_model.reset_incremental_jacobians(torch.Tensor(P0).unsqueeze(0))
    u_torch = torch.zeros(1, 2, 3)
    u_torch[:, 0, :] = torch.Tensor(u.gyro)
    u_torch[:, 1, :] = torch.Tensor(u.accel)

    X_0_t = SE23.Exp(torch.Tensor(([1, 2, 3, 4, 5, 6, 7, 8, 9])).reshape(1, 9, 1))

    # Do both dead reckoning and preintegration
    for i in range(100):
        rmi.increment(u, dt)
        X_0_t = dive_preint_model.evaluate(X_0_t, u_torch, dt)

    # Apply the rmi to the state
    ekf.process_model = preint_model
    x_pre = ekf.predict(x0.copy(), rmi, dt)

    # Compare the results
    assert torch.allclose(X_0_t, torch.Tensor(x_pre.state.pose), atol=1e-6), "Preintegration pose equivalency failed"
    assert torch.allclose(dive_preint_model.Q_ij, torch.Tensor(rmi.covariance), atol=1e-6), "Preint covariance equivalency failed"
    assert torch.allclose(dive_preint_model.B_ij, torch.Tensor(rmi.bias_jacobian), atol=1e-6), "Preint bias jacobian equivalency failed"
    assert torch.allclose(dive_preint_model.P_j, torch.Tensor(x_pre.covariance), atol=1e-5), "Preintegration state uncertainty failed"

    print("All tests passed!")

if __name__ == "__main__":
    test_preintegration_functions()
    test_imu_preintegration_equivalence("right")
