import torch

from pymlg.torch import SO3
from pymlg.torch import utils

def unit_vec_rodrigues(a : torch.Tensor, b : torch.Tensor):
    """
    returns a rotation matrix R that rotates unit vector a onto unit vector b

    Parameters
    ----------
    a : torch.Tensor
        axis batch of rotation vectors : torch.Tensor with shape (N, 3)
    b : torch.Tensor
        axis batch of rotation vectors : torch.Tensor with shape (N, 3)

    Returns
    -------
    C : torch.Tensor
        Batched rotation matrix : torch.Tensor with shape (N, 3, 3)
    """

    alpha = torch.cross(input = a, other = b)
    beta = torch.bmm(a.unsqueeze(2).transpose(1, 2), b.unsqueeze(2))

    alpha_cross = SO3.wedge(alpha.unsqueeze(2))

    C = torch.eye(3, 3, device=a.device).unsqueeze(0) + alpha_cross + (alpha_cross @ alpha_cross) * (1 / (1 + beta))

    return C

def calculate_rotation_error(c_t, c_hat):
    """
    calculates the geodesic distance and corresponding rotation error between two [N, 3, 3] rotation histories

    individual error computation is e = Log(C_hat @ C_t.T)

    returns axis [N, 3, 1] set of rotation vectors
    """

    e_C = SO3.Log((c_hat @ c_t.transpose(1, 2)))

    return e_C


def form_time_machine(N, dt):
    """
    forms axis "time machine" matrix as axis SE_2(3) helper - section 9.4.7 (http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser22.pdf)
    to isolate the input as axis member of SE_2(3)

    Parameters
    ----------
    N : float
        Desired batch size : float
    dt : float
        The time interval :math:`\Delta t` between the two states : float

    Returns
    -------
    delta_{k} : torch.Tensor
        Batched time machine matrix : torch.Tensor with shape (N, 5, 5)
    """

    delta_k = utils.batch_eye(N, 5, 5)

    delta_k[:, 3, 4] = dt

    return delta_k


def form_N_matrix(phi):
    """
    forms the N matrix used to create the state propogation matrix - refer to Compact IMU Kinematics document [TODO: some link?]

    Parameters
    ----------
    phi : torch.Tensor
        axis batch of rotation vectors : torch.Tensor with shape (N, 3)

    Returns
    -------
    N : torch.Tensor
        Batched time machine matrix : torch.Tensor with shape (N, 3, 3)
    """

    # enforce dimensionality by squeezing beyond first dimension
    if phi.dim() > 2:
        phi = phi.squeeze(dim=2)

    if torch.linalg.norm(phi) < SO3._small_angle_tol:
        return torch.eye(3, 3).unsqueeze(0)

    # axis representation
    axis = (phi / torch.linalg.norm(phi, dim=1).unsqueeze(1)).reshape(-1, 3, 1)

    # norm representation
    phi_scalar = torch.linalg.norm(phi, dim=1).unsqueeze(1)

    N = (
        axis @ axis.transpose(1, 2)
        + 2
        * (1 / phi_scalar - torch.sin(phi_scalar) / (phi_scalar**2)).unsqueeze(2)
        * SO3.wedge(axis)
        + 2
        * ((torch.cos(phi_scalar) - 1) / (phi_scalar**2)).unsqueeze(2)
        * (SO3.wedge(axis) @ SO3.wedge(axis))
    )

    return N

# util for computing the derivative of C -> C_gamma transformation
def c_c_gamma_mapping_derivative(euler):
    """
    euler is expected to be a (1, 2, 3) euler decomposition of a rotation matrix
    """

    H_z = torch.zeros(3, 3, device=euler.device).unsqueeze(0)

    H_z[:, 2, 0] = torch.cos(euler[:, 2]) * torch.tan(euler[:, 1])
    H_z[:, 2, 1] = torch.sin(euler[:, 2]) * torch.tan(euler[:, 1])
    H_z[:, 2, 2] = 1

    return H_z

# utils for decomposition of rotation matricies

def c_3(phi):
    zero = phi.new_zeros(phi.shape[0]).unsqueeze(1)
    ones = phi.new_ones(phi.shape[0]).unsqueeze(1)

    C_gamma = torch.stack(
        (
            torch.cos(phi[:, 2]),
            torch.sin(phi[:, 2]),
            zero,
            -torch.sin(phi[:, 2]),
            torch.cos(phi[:, 2]),
            zero,
            zero,
            zero,
            ones,
        ),
        1,
    ).view(phi.shape[0], 3, 3)

    return C_gamma


def c_2(phi):
    zero = phi.new_zeros(phi.shape[0]).unsqueeze(1)
    ones = phi.new_ones(phi.shape[0]).unsqueeze(1)
    C_beta = torch.stack(
        (
            torch.cos(phi[:, 1]),
            zero,
            -torch.sin(phi[:, 1]),
            zero,
            ones,
            zero,
            torch.sin(phi[:, 1]),
            zero,
            torch.cos(phi[:, 1]),
        ),
        1,
    ).view(phi.shape[0], 3, 3)

    return C_beta


def c_1(phi):
    zero = phi.new_zeros(phi.shape[0]).unsqueeze(1)
    ones = phi.new_ones(phi.shape[0]).unsqueeze(1)
    C_alpha = torch.stack(
        (
            ones,
            zero,
            zero,
            zero,
            torch.cos(phi[:, 0]),
            torch.sin(phi[:, 0]),
            zero,
            -torch.sin(phi[:, 0]),
            torch.cos(phi[:, 0]),
        ),
        1,
    ).view(phi.shape[0], 3, 3)

    return C_alpha
