import math
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import truncnorm

EPSILON = 1e-8

def LERP(translation_0, translation_1, timestep):
    translation_t = (1 - timestep) * translation_0 + timestep * translation_1  # [num_cluster*17, 3]
    return translation_t


def SLERP(quaternion_0, quaternion_1, timestep):
    dot_product = torch.einsum("ab,ab->a", quaternion_0, quaternion_1).unsqueeze(1)  # [num_cluster*17, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
    omega = torch.acos(dot_product)  # [num_cluster*17, 1]
    sin_omega = torch.sin(omega)  # [num_cluster*17, 1]
    omega_t = omega * timestep  # [num_cluster*17, 1]
    sin_omega_t = torch.sin(omega_t)  # [num_cluster*17, 1]
    omega_1_t = omega * (1 - timestep)  # [num_cluster*17, 1]
    sin_omega_1_t = torch.sin(omega_1_t)  # [num_cluster*17, 1]
    quaternion_t = sin_omega_1_t /sin_omega * quaternion_0 + sin_omega_t /sin_omega * quaternion_1  # [num_cluster*17, 4]

    # NOTE: check numerical stability when omega=0
    quaternion_t = torch.where(
        omega < EPSILON,
        LERP(quaternion_0, quaternion_1, timestep), # Do LERP if q0 and q1 are close
        quaternion_t,
    )

    # TODO: should we guarantee this?
    mask = quaternion_t[..., 0] < 0
    quaternion_t[mask] *= -1
    return quaternion_t


def SLERP_derivative(quaternion_0, quaternion_1, timestep):
    dot_product = torch.sum(quaternion_0 * quaternion_1, dim=-1).unsqueeze(1)  # [num_cluster*17, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability

    omega = torch.acos(dot_product)  # [num_cluster*17, 1]
    omega_t = omega * timestep  # [num_cluster*17, 1]
    omega_1_t = omega * (1 - timestep)  # [num_cluster*17, 1]
    sin_omega = torch.sin(omega)  # [num_cluster*17, 1]

    cos_omega_t = torch.cos(omega_t)  # [num_cluster*17, 1]
    cos_omega_1_t = torch.cos(omega_1_t)  # [num_cluster*17, 1]

    derivative = omega / sin_omega * (-cos_omega_1_t * quaternion_0 + cos_omega_t * quaternion_1)
    return derivative


class Linear(nn.Linear):
    def __init__(self, in_dim, out_dim, bias=True, init="default", init_fn=None,):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
        return


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}

def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat

_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: torch.Tensor):
    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]] = rot

    K = [
        [
            R11 + R22 + R33,
            R32 - R23,
            R13 - R31,
            R21 - R12,
        ],
        [
            R32 - R23,
            R11 - R22 - R33,
            R12 + R21,
            R13 + R31,
        ],
        [
            R13 - R31,
            R12 + R21,
            R22 - R11 - R33,
            R23 + R32,
        ],
        [
            R21 - R12,
            R13 + R31,
            R23 + R32,
            R33 - R11 - R22,
        ],
    ]

    K = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in K], dim=-2)

    _, vectors = torch.linalg.eigh(K)
    quaternion = vectors[..., -1]
    
    mask = quaternion[..., 0] < 0
    quaternion[mask] *= -1
    return quaternion

    
def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)
    ], dim=-1).reshape(-1, 3, 3)
    
    return R


def rotation_matrix_to_quaternion(R):
    device = R.device
    # R shape: (batch_size, 3, 3)
    batch_size = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    trace_mask = trace > 0
    s = torch.where(trace_mask, torch.sqrt(trace + 1.0) * 2, torch.tensor(0.0, device=device))

    # Precompute
    qw = 0.25 * s
    qx = torch.where(trace_mask, (R[:, 2, 1] - R[:, 1, 2]) / s, torch.tensor(0.0, device=device))
    qy = torch.where(trace_mask, (R[:, 0, 2] - R[:, 2, 0]) / s, torch.tensor(0.0, device=device))
    qz = torch.where(trace_mask, (R[:, 1, 0] - R[:, 0, 1]) / s, torch.tensor(0.0, device=device))

    cond1 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    cond2 = R[:, 1, 1] > R[:, 2, 2]
    
    S1 = torch.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) * 2
    S2 = torch.sqrt(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]) * 2
    S3 = torch.sqrt(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]) * 2

    qx = torch.where(cond1 & ~trace_mask, 0.25 * S1, qx)
    qy = torch.where(cond2 & ~trace_mask & ~cond1, 0.25 * S2, qy)
    qz = torch.where(~trace_mask & ~cond1 & ~cond2, 0.25 * S3, qz)

    qw = torch.where(cond1 & ~trace_mask, (R[:, 2, 1] - R[:, 1, 2]) / S1, qw)
    qw = torch.where(cond2 & ~trace_mask & ~cond1, (R[:, 0, 2] - R[:, 2, 0]) / S2, qw)
    qw = torch.where(~trace_mask & ~cond1 & ~cond2, (R[:, 1, 0] - R[:, 0, 1]) / S3, qw)

    qx = torch.where(cond2 & ~trace_mask & ~cond1, (R[:, 0, 1] + R[:, 1, 0]) / S2, qx)
    qx = torch.where(~trace_mask & ~cond1 & ~cond2, (R[:, 0, 2] + R[:, 2, 0]) / S3, qx)
    
    qy = torch.where(cond1 & ~trace_mask, (R[:, 0, 1] + R[:, 1, 0]) / S1, qy)
    qy = torch.where(~trace_mask & ~cond1 & ~cond2, (R[:, 1, 2] + R[:, 2, 1]) / S3, qy)
    
    qz = torch.where(cond1 & ~trace_mask, (R[:, 0, 2] + R[:, 2, 0]) / S1, qz)
    qz = torch.where(cond2 & ~trace_mask & ~cond1, (R[:, 1, 2] + R[:, 2, 1]) / S2, qz)

    quaternion = torch.stack([qw, qx, qy, qz], dim=1)

    mask = quaternion[..., 0] < 0
    quaternion[mask] *= -1
    return quaternion


# Function to perform quaternion-and-vec multiplication for batches
def quaternion_multiply_by_vec_left(quat, vec):
    """
        quat: [N, 4]
        vec: [N, 3] ==> (1, x, y, z)
        rotate with [1, vec] first, then rotate with quat
    """
    w1, x1, y1, z1 = quat.unbind(-1)
    w2 = 1
    x2, y2, z2 = vec.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


# Function to perform quaternion-and-vec multiplication for batches
def quaternion_multiply_by_vec_right(quat, vec):
    """
        quat: [N, 4]
        vec: [N, 3] ==> (1, x, y, z)
        quat: [N, 4]
        vec: [N, 3] ==> (1, x, y, z)
        rotate with quat first, then rotate with [1, vec]
    """
    w2, x2, y2, z2 = quat.unbind(-1)
    w1 = 1
    x1, y1, z1 = vec.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


# Function to get the conjugate of a batch of quaternions
def quaternions_conjugate(qs):
    qs_conj = qs.clone()
    qs_conj[..., 1:] = -qs_conj[..., 1:]
    return qs_conj


# Function to perform quaternion multiplication for batches
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


# Function to normalize a batch of quaternions
def normalize_quaternions(qs):
    norms = torch.norm(qs, dim=-1, keepdim=True)
    return qs / norms


# Function to rotate a batch of points using a batch of quaternions
def rotate_points_with_quaternion(points, quaternions):
    """
        points: ..., 3
        quaternions: ..., 4
    """
    # Normalize the quaternions
    quaternions = normalize_quaternions(quaternions)

    # Convert points to quaternions (..., 0, v_x, v_y, v_z)
    batch_size = points.size(0)
    device = points.device
    neo_shape = points.shape[:-1] + (1,)
    p = torch.cat([torch.zeros(neo_shape).to(device), points], dim=-1)  # Add zero as the scalar part
    
    # Calculate the rotated points
    q_conjugate = quaternions_conjugate(quaternions)
    
    # NOTE: this is passive rotation
    rotated_p = quaternion_multiply(quaternions, quaternion_multiply(p, q_conjugate))

    # Extract the vector part (v_x, v_y, v_z)
    rotated_points = rotated_p[..., 1:]
    return rotated_points


def move_molecule(initial_positions, translation_0, translation_t, quaternion_0, quaternion_t, atom2molecule):
    quaternion_0 = quaternion_0[atom2molecule]  # [N, 4]
    quaternion_0_conj = quaternions_conjugate(quaternion_0)  # [N, 4]
    translation_0 = translation_0[atom2molecule]  # [N, 3]

    quaternion_t = quaternion_t[atom2molecule]  # [N, 4]
    translation_t = translation_t[atom2molecule] # [N, 3]

    positions = initial_positions - translation_0
    
    # q_1_to_2 = quaternion_multiply(quaternion_t, quaternion_0_conj)
    # positions = rotate_points_with_quaternion(positions, q_1_to_2)

    positions = rotate_points_with_quaternion(positions, quaternion_0_conj)
    positions = rotate_points_with_quaternion(positions, quaternion_t)

    positions = positions + translation_t
    return positions
