import torch
from pymlg.torch import SO3
from filtering.filtering_utils import unit_vec_rodrigues


def loss_rotation(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    given two rotation vectors resolved in the same frame, compute the rot. matrix between them and return it's frobenius norm
    """

    pred = pred / torch.norm(pred, dim=1).unsqueeze(1)

    C = unit_vec_rodrigues(pred.view(-1, 3), targ.view(-1, 3))

    # C = SO3.Exp(pred) @ SO3.Exp(targ).transpose(1, 2)

    # take the trace of I - C
    return torch.mean(
        (torch.eye(3, 3, device=device).unsqueeze(0) - C)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )

def clip_abs(tens : torch.Tensor, eps=1e-7):
    sign = tens.sign()
    tens = tens.abs_().clamp_(eps, None)
    tens *= sign

    return tens

def gen_cov_with_activation_cholesky(output : torch.Tensor, device):
    """
    generate a covariance matrix based on a [N, 6] output vector, using exp() activations for the diagonal elements and tanh() activations
    for the off-diagonal elements
    """

    mat = torch.zeros(output.shape[0], 3, 3, device=output.device)
    mat[:, 0, 0] = torch.exp(output[:, 0])
    mat[:, 1, 1] = torch.exp(output[:, 1])
    mat[:, 2, 2] = torch.exp(output[:, 2])
    mat[:, 1, 0] = torch.tanh(output[:, 3])
    mat[:, 2, 0] = torch.tanh(output[:, 4])
    mat[:, 2, 1] = torch.tanh(output[:, 5])

    return mat @ mat.transpose(1, 2)

def gen_cov_diag_only(p : torch.Tensor, device):
    """
    Args:
        pred_cov [n x 3] : xx, yy, zz, rho_xy, rho_xz, rho_yz
    Returns:
        cov [n x 3 x 3] : full covariance
    """
    N = p.shape[0]
    # activate rhos as in https://arxiv.org/pdf/1910.14215.pdf
    # alpha = 0.05
    # eps = 1e-3  # "force the Pearson correlation coefficients to not get too close to 1"
    # rho_xy = (1 - eps) * torch.tanh(alpha * p[:, 3])
    # rho_xz = (1 - eps) * torch.tanh(alpha * p[:, 4])
    # rho_yz = (1 - eps) * torch.tanh(alpha * p[:, 5])

    covf = torch.zeros((N, 9), device=device)

    # on diagonal terms
    covf[:, 0] = torch.exp(2 * p[:, 0])
    covf[:, 4] = torch.exp(2 * p[:, 1])
    covf[:, 8] = torch.exp(2 * p[:, 2])

    return covf.reshape((N, 3, 3))

def gen_cov_tlio_form(p : torch.Tensor, device):
        """
        Args:
            pred_cov [n x 6] : xx, yy, zz, rho_xy, rho_xz, rho_yz
        Returns:
            cov [n x 3 x 3] : full covariance
        """
        N = p.shape[0]
        # activate rhos as in https://arxiv.org/pdf/1910.14215.pdf
        alpha = 0.05
        eps = 1e-3  # "force the Pearson correlation coefficients to not get too close to 1"
        rho_xy = (1 - eps) * torch.tanh(alpha * p[:, 3])
        rho_xz = (1 - eps) * torch.tanh(alpha * p[:, 4])
        rho_yz = (1 - eps) * torch.tanh(alpha * p[:, 5])

        covf = torch.zeros((N, 9), device=device)

        c1 = rho_xy * torch.sqrt(covf[:, 0] * covf[:, 4]) 
        c2 = rho_xz * torch.sqrt(covf[:, 0] * covf[:, 8])
        c3 = rho_yz * torch.sqrt(covf[:, 4] * covf[:, 8])

        # on diagonal terms
        covf[:, 0] = torch.exp(2 * p[:, 0])
        covf[:, 4] = torch.exp(2 * p[:, 1])
        covf[:, 8] = torch.exp(2 * p[:, 2])

        # off diagonal terms
        covf[:, 1] = c1  # xy
        covf[:, 2] = c2  # xz
        covf[:, 5] = c3  # yz
        
        # symmetry
        covf[:, 3] = c1  # xy
        covf[:, 6] = c2  # xz
        covf[:, 7] = c3 # yz

        return covf.reshape((N, 3, 3))

def loss_geodesic(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    given two rotation vectors resolved in the same frame, compute the geodesic distance between them on the unit circle

    pred : torch.Tensor with shape [N, 3]
    targ : torch.Tensor with shape [N, 1, 3]
    """

    pred = pred / torch.norm(pred, dim=1).unsqueeze(1)
    
    # pred_targ_dot = torch.bmm(targ, pred.unsqueeze(2))

    # pred_targ_dot = torch.clamp(input = pred_targ_dot, min = -1 + 1e-7, max = 1 - 1e-7)

    return torch.mean(torch.arctan2(torch.norm(torch.cross(pred, targ.squeeze(1)), dim=1), torch.bmm(targ, pred.unsqueeze(2)).view(-1)))# torch.arccos(pred_targ_dot))

def loss_mse(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    given two rotation vectors resolved in the same frame, compute the geodesic distance between them on the unit circle

    pred : torch.Tensor with shape [N, 3]
    targ : torch.Tensor with shape [N, 1, 3]
    """

    return torch.mean((pred - targ.squeeze(1)).pow(2))

def loss_NLL_diag(pred : torch.Tensor, pred_cov : torch.Tensor, targ : torch.Tensor, device):
    # construct positive semidefinite matrix from
    pred_cov = clip_abs(pred_cov)
    pred_cov = gen_cov_diag_only(p = pred_cov, device = device)

    # change to desired shape
    pred = pred.view(-1, 3, 1)
    targ = targ.view(-1, 3, 1)

    # return loss_scalar
    loss = torch.mean(.5 * ((targ-pred).transpose(1, 2) @ torch.linalg.inv(pred_cov) @ (targ - pred)).view(-1) + .5 * torch.logdet(pred_cov))
    return loss

def loss_NLL_diag_TLIO(pred : torch.Tensor, pred_cov : torch.Tensor, targ : torch.Tensor, device):
    
    # enforce dimensionality
    pred = pred.view(-1, 3)
    targ = targ.view(-1, 3)
    pred_cov = pred_cov.view(-1, 3) + 1e-7
    loss = torch.mean(((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov)) + pred_cov)
    return loss

def loss_NLL(pred : torch.Tensor, pred_cov : torch.Tensor, targ : torch.Tensor, device):

    # construct positive semidefinite matrix from
    pred_cov = clip_abs(pred_cov)
    pred_cov = gen_cov_tlio_form(p = pred_cov, device = device)

    # change to desired shape
    pred = pred.view(-1, 3, 1)
    targ = targ.view(-1, 3, 1)

    # return loss_scalar
    loss = torch.mean(.5 * ((targ-pred).transpose(1, 2) @ torch.linalg.inv(pred_cov) @ (targ - pred)).view(-1) + .5 * torch.logdet(pred_cov))
    return loss

def loss_mse_into_NLL(pred : torch.Tensor, pred_cov : torch.Tensor, targ : torch.Tensor, device, epoch):
    if epoch < 10:
        return loss_mse(pred, targ, device)
    else:
        return loss_NLL_diag_TLIO(pred = pred, pred_cov = pred_cov, targ = targ, device=device)

def loss_geodesic_into_NLL(pred : torch.Tensor, pred_cov : torch.Tensor, targ : torch.Tensor, device, epoch):
    if epoch < 10:
        return loss_geodesic(pred, targ, device)
    else:
        return loss_NLL(pred = pred, pred_cov = pred_cov, targ = targ, device=device)

def loss_geodesic_sample(pred : torch.Tensor, targ : torch.Tensor, device):
    """
    given two rotation vectors resolved in the same frame, compute the geodesic distance between them on the unit circle
    """

    pred = pred / torch.norm(pred, dim=1).unsqueeze(1)

    return torch.arctan2(torch.norm(torch.cross(pred, targ.squeeze(1)), dim=1), torch.bmm(targ, pred.unsqueeze(2)).view(-1))

def loss_rotation_sample(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    given two rotation vectors resolved in the same frame, compute the rot. matrix between them and return it's frobenius norm
    """

    pred = pred / torch.norm(pred, dim=1).unsqueeze(1)

    C = unit_vec_rodrigues(pred.view(-1, 3), targ.view(-1, 3))

    # C = SO3.Exp(pred) @ SO3.Exp(targ).transpose(1, 2)

    # take the trace of I - C
    return (
        (torch.eye(3, 3, device=device).unsqueeze(0) - C)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )


def loss_frobenius(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    given two unit vectors resolved in the same frame, compute the rot. matrix between them and return it's frobenius norm
    """

    alpha = torch.cross(input=pred, other=targ)
    beta = torch.bmm(pred.unsqueeze(2).transpose(1, 2), targ.unsqueeze(2))

    alpha_cross = SO3.wedge(alpha.unsqueeze(2))

    C = alpha_cross + (alpha_cross @ alpha_cross) * (
        1 / (1 + beta)
    )  # torch.eye(3, 3, device = device).unsqueeze(0)

    return torch.mean(torch.norm(SO3.Log(C), dim=1))


def loss_frobenius_v2(pred: torch.Tensor, targ: torch.Tensor, device):
    """
    secondary method of computing the frobenius norm of the rotation matrix between two unit vectors resolved in the same frame by forming an orthogonal
    basis from the vectors + plane, and then projecting the 2D 3-axis transformation on the plane to the stationary resolved frame
    """

    # generate rotation matrix on the plane defined having the normal (a x b) by substituing cos(theta) = a /dot b and sin(theta) = ||a x b||

    cos_theta = torch.bmm(pred.unsqueeze(2).transpose(1, 2), targ.unsqueeze(2)).squeeze(
        2
    )

    sin_theta = torch.norm(torch.cross(input=pred, other=targ), dim=1).unsqueeze(1)

    zero = cos_theta.new_zeros(cos_theta.shape[0]).view(-1, 1)
    ones = cos_theta.new_ones(cos_theta.shape[0]).view(-1, 1)

    G = torch.stack(
        (
            cos_theta,
            -sin_theta,
            zero,
            sin_theta,
            cos_theta,
            zero,
            zero,
            zero,
            ones,
        ),
        1,
    ).view(cos_theta.shape[0], 3, 3)

    u_proj = pred

    v_rejection = (targ - cos_theta.squeeze(1) @ u_proj) / torch.norm(
        targ - cos_theta.squeeze(1) @ pred, dim=1
    ).unsqueeze(1)

    w_cross = torch.cross(input=targ, other=pred)

    F = torch.linalg.inv(
        torch.cat(
            (u_proj.unsqueeze(2), v_rejection.unsqueeze(2), w_cross.unsqueeze(2)), dim=2
        )
    )

    U = torch.linalg.inv(F) @ G @ F  # this defines b = U @ a

    return torch.mean(
        torch.linalg.matrix_norm(U - torch.eye(3, 3, device=device).unsqueeze(0))
    )

    # return torch.mean(torch.norm(SO3.Log(U), dim=1))
