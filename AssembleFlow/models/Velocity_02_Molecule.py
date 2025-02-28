import math
from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import softmax
from .utils import Linear, ipa_point_weights_init_
from .utils import quaternion_multiply, quat_to_rot, rotate_points_with_quaternion, quaternions_conjugate


EPS = EPSILON = 1e-8


class InvariantPointAttention(nn.Module):
    def __init__(self, ipa_config):
        super(InvariantPointAttention, self).__init__()
        self._ipa_config = ipa_config

        self.s_dim = ipa_config.s_dim
        self.z_dim = ipa_config.z_dim
        self.qk_dim = ipa_config.qk_dim
        self.v_dim = ipa_config.v_dim
        self.num_heads = ipa_config.num_heads

        self.linear_q_scalar = Linear(self.s_dim, self.num_heads * self.s_dim, bias=False)
        self.linear_k_scalar = Linear(self.s_dim, self.num_heads * self.s_dim, bias=False)
        self.linear_v_scalar = Linear(self.s_dim, self.num_heads * self.s_dim, bias=False)

        self.linear_q_vector = Linear(self.s_dim, self.num_heads * self.qk_dim * 3, bias=False)
        self.linear_k_vector = Linear(self.s_dim, self.num_heads * self.qk_dim * 3, bias=False)
        self.linear_v_vector = Linear(self.s_dim, self.num_heads * self.v_dim * 3, bias=False)

        self.linear_b = Linear(self.z_dim, self.num_heads)
        self.down_z = Linear(self.z_dim, self.z_dim // 4)

        self.head_weights = nn.Parameter(torch.zeros((self.num_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.z_dim // 4 + self.s_dim + self.v_dim * 4
        self.linear_out = Linear(self.num_heads * concat_out_dim, self.s_dim, init="final")

        self.softplus = nn.Softplus()
        return

    def forward(self, s, z, edge_index, quaternion, translation):
        """
        s: [N, d]
        z: [E, d]
        """
        index_i, index_j = edge_index
        N = s.shape[0]

        #######################################
        # Generate scalar and point activations
        #######################################
        q_scalar = self.linear_q_scalar(s)  # [N, num_head * d]
        k_scalar = self.linear_k_scalar(s)  # [N, num_head * d]
        v_scalar = self.linear_v_scalar(s)  # [N, num_head * d]

        q_scalar = q_scalar.view((N, self.num_heads, -1)) # [N, num_head, d]
        k_scalar = k_scalar.view((N, self.num_heads, -1))  # [N, num_head, d]
        v_scalar = v_scalar.view((N, self.num_heads, -1))  # [N, num_head, d]

        q_vector = self.linear_q_vector(s)  # [N, num_head * qk_dim * 3]
        k_vector = self.linear_k_vector(s)  # [N, num_head * qk_dim * 3]
        v_vector = self.linear_v_vector(s)  # [N, num_head * v_dim * 3]

        q_vector = torch.split(q_vector, q_vector.shape[-1] // 3, dim=-1)
        k_vector = torch.split(k_vector, k_vector.shape[-1] // 3, dim=-1)
        v_vector = torch.split(v_vector, v_vector.shape[-1] // 3, dim=-1)

        q_vector = torch.stack(q_vector, dim=-1)  # [N, num_head * qk_dim, 3]
        k_vector = torch.stack(k_vector, dim=-1)  # [N, num_head * qk_dim, 3]
        v_vector = torch.stack(v_vector, dim=-1)  # [N, num_head * v_dim, 3]

        q_vector = rotate_points_with_quaternion(q_vector, quaternion.unsqueeze(1)) + translation.unsqueeze(1)
        k_vector = rotate_points_with_quaternion(k_vector, quaternion.unsqueeze(1)) + translation.unsqueeze(1)
        v_vector = rotate_points_with_quaternion(v_vector, quaternion.unsqueeze(1)) + translation.unsqueeze(1)

        q_vector = q_vector.view((N, self.num_heads, self.qk_dim, 3))  # [N, num_head, qk_dim, 3]
        k_vector = k_vector.view((N, self.num_heads, self.qk_dim, 3))  # [N, num_head, qk_dim, 3]
        v_vector = v_vector.view((N, self.num_heads, self.v_dim, 3))   # [N, num_head, v_dim, 3]

        ##########################
        # Compute attention scores
        ##########################
        b = self.linear_b(z)  # [E, num_head]

        attn_scalar = k_scalar[index_i] * q_scalar[index_j]  # [E, num_head, d]
        attn_scalar = attn_scalar.sum(-1)  # [E, num_head]
        attn_scalar = attn_scalar * math.sqrt(1.0 / 3 * (self.qk_dim))  # [E, num_head, qk_dim]
        attn_scalar = attn_scalar + math.sqrt(1.0 / 3) * b  # [E, num_head, qk_dim]

        vector_displacement = q_vector[index_i] - k_vector[index_j]  # [E, num_head, qk_dim, 3]
        vector_att = vector_displacement**2  # [E, num_head, qk_dim, 3]

        vector_att = sum(torch.unbind(vector_att, dim=-1))  # [E, num_head, qk_dim]
        head_weights = self.softplus(self.head_weights)  # [num_head]
        head_weights = head_weights.unsqueeze(0).unsqueeze(2)  # [1, num_head, 1]
        vector_att = vector_att * head_weights * math.sqrt(1.0 / (3 * (self.qk_dim * 9.0 / 2)))  # [E, num_head, qk_dim]
        
        vector_att = torch.sum(vector_att, dim=-1) * (-0.5)  # [E, num_head]
        attn_scalar = attn_scalar + vector_att
        attn_scalar = softmax(attn_scalar, index=index_i, dim=0)  # [E, num_head]

        ################
        # Compute output
        ################
        output_scalar_edge = attn_scalar.unsqueeze(2) * v_scalar[index_i]  # [E, num_head, d]
        output_scalar = scatter(output_scalar_edge, index_i, dim=0, dim_size=N, reduce="sum")  # [N, num_head, d]
        output_scalar = output_scalar.reshape(N, -1)  # [N, num_head * d]

        output_vector_edge = attn_scalar.unsqueeze(2).unsqueeze(3) * v_vector[index_i]  # [E, num_head, v_dim, 3]
        output_vector = scatter(output_vector_edge, index=index_i, dim_size=N, dim=0)  # [N, num_head, v_dim, 3]

        output_vector = output_vector - translation.unsqueeze(1).unsqueeze(1)  # [N, num_head, v_dim, 3]
        # NOTE: conjugate of quaternion is inverse rotation
        output_vector = rotate_points_with_quaternion(output_vector, quaternions_conjugate(quaternion).unsqueeze(1).unsqueeze(1))  # [N, num_head, v_dim, 3]
        
        output_vector_dists = torch.sqrt(torch.sum(output_vector**2, dim=-1) + EPS)  # [N, num_head, v_dim]
        output_vector_norm_feats = output_vector_dists.view(N, -1)  # [N, num_head * v_dim]

        output_vector = output_vector.reshape(N, -1, 3)  # [N, num_head * v_dim, 3]

        pair_z = self.down_z(z).to(dtype=attn_scalar.dtype)  # [E, z_dim//4]
        output_pair_edge = attn_scalar.unsqueeze(2) * pair_z.unsqueeze(1)  # [E, num_head, z_dim//4]
        output_pair = scatter(output_pair_edge, index=index_i, dim=0, dim_size=N, reduce="sum")  # [N, num_head, z_dim//4]
        output_pair = output_pair.reshape(N, -1)  # [N, num_head * z_dim//4]

        # [N, num_head * d], [N, num_head * v_dim, 3], [N, num_head * v_dim], [N, num_head * z_dim//4]
        output_feats = [output_scalar, *torch.unbind(output_vector, dim=-1), output_vector_norm_feats, output_pair]

        s = self.linear_out(torch.cat(output_feats, dim=-1).to(dtype=z.dtype))  # [N, d]
        return s


class NodeTransition(nn.Module):
    def __init__(self, c):
        super(NodeTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)
        return s


class EdgeTransition(nn.Module):
    def __init__(self, node_attr_size, edge_embed_in, edge_embed_out, num_layers=2, node_dilation=2,):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_attr_size // node_dilation
        self.initial_embed = Linear(node_attr_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)
        return

    def forward(self, node_embed, edge_embed, edge_index):
        index_i, index_j = edge_index
        node_embed = self.initial_embed(node_embed)
        edge_embed = torch.cat([edge_embed, node_embed[index_i], node_embed[index_j]], axis=-1)

        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        return edge_embed


# NOTE: Invariant Point Attention
class Velocity_Molecule(nn.Module):
    def __init__(self, hidden_dim, ipa_config, flow_matcher):
        super(Velocity_Molecule, self).__init__()
        self.trunk = nn.ModuleDict()

        self.ipa_config = ipa_config
        self.hidden_dim = hidden_dim
        
        coordinate_scaling = 0.1
        self.scale_pos = lambda x: x * coordinate_scaling
        self.unscale_pos = lambda x: x / coordinate_scaling

        for b in range(self.ipa_config.num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_config)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(ipa_config.s_dim)
            self.trunk[f"skip_embed_{b}"] = Linear(self.hidden_dim, self.hidden_dim, init="final")
            
            self.trunk[f"node_transition_{b}"] = NodeTransition(ipa_config.s_dim)
            self.trunk[f"bb_update_{b}"] = Linear(ipa_config.s_dim, 6, init="final")

            if b < self.ipa_config.num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_attr_size=ipa_config.s_dim,
                    edge_embed_in=ipa_config.z_dim,
                    edge_embed_out=ipa_config.z_dim,
                )
        return

    def forward(self, node_attr, edge_attr, edge_index, node2graph, time_attr, equivariant_basis, input_feats):
        init_translation, init_quaternion = input_feats
        curr_translation = torch.clone(init_translation)
        curr_quaternion = torch.clone(init_quaternion)

        curr_translation = self.scale_pos(curr_translation)

        node_embed = node_attr
        edge_embed = edge_attr

        for b in range(self.ipa_config.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, edge_index, quaternion=curr_quaternion, translation=curr_translation)  # [N, d]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)  # [N, d]

            node_embed = node_embed + self.trunk[f"skip_embed_{b}"](node_attr)  # [N, d]

            node_embed = self.trunk[f"node_transition_{b}"](node_embed)  # [N, d]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed)  # [N, 6]

            if b < self.ipa_config.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed, edge_index)

            #### Update quaternion and translation #####
            quaternion_update_vec, translation_update = rigid_update.chunk(2, dim = -1)  # [N, 3], [N, 3]

            curr_translation = curr_translation + translation_update

            curr_quaternion_02 = curr_quaternion.clone()
            
            quaternion_update = F.pad(quaternion_update_vec, (1, 0), value = 1.)  # [N, 4]
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)  # [N, 4]
            curr_quaternion = quaternion_multiply(quaternion_update, curr_quaternion)  # [N, 4]
            curr_quaternion = curr_quaternion / (torch.linalg.norm(curr_quaternion, dim=-1, keepdim=True) + EPS)  # [N, 4]

        curr_translation = self.unscale_pos(curr_translation)

        return curr_translation, curr_quaternion
