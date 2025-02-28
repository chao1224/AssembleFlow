from tqdm import tqdm
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import radius_graph, radius
from .Velocity_01_Atom import Velocity_Atom
from .Velocity_02_Molecule import Velocity_Molecule
from .PaiNN import PaiNN
from .utils import rot_to_quat, LERP, SLERP, SLERP_derivative, move_molecule


EPSILON = 1e-8


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        # print("activation in MultiLayerPerceptron", self.activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


def coord2basis(pos_intra, pos_center, index_intra, index_center):
    coord_diff = pos_intra[index_intra] - pos_center[index_center]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_cross = torch.cross(pos_intra[index_intra], pos_center[index_center])

    norm = torch.sqrt(radial) + EPSILON
    coord_diff = coord_diff / norm
    cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1)) + EPSILON
    coord_cross = coord_cross / cross_norm

    coord_vertical = torch.cross(coord_diff, coord_cross)

    return coord_diff, coord_cross, coord_vertical


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AssembleFlow(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim, cutoff, cluster_cutoff, node_class, args,
        num_timesteps, anneal_power=0,
        short_cut=False, concat_hidden=False):

        super(AssembleFlow, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.cluster_cutoff = cluster_cutoff
        self.anneal_power = anneal_power
        self.alpha_rotation = args.alpha_rotation
        self.alpha_translation = args.alpha_translation

        self.model_3d = args.model_3d
        if self.model_3d == "PaiNN":
            self.intra_model = PaiNN(
                n_atom_basis=args.emb_dim,  # default is 64
                n_interactions=args.PaiNN_n_interactions,
                n_rbf=args.PaiNN_n_rbf,
                cutoff=self.cutoff,
                max_z=node_class,
                n_out=1, # This won't take effect
                readout=args.PaiNN_readout,
            )

        self.num_timesteps = num_timesteps

        self.time_embed_dim = self.emb_dim * 2
        self.time_embed = nn.Sequential(
            nn.Linear(self.emb_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # https://github.com/atong01/conditional-flow-matching/blob/ec4da0846ddaf77e8406ad2fd592a6f0404ce5ae/torchcfm/models/unet/unet.py#L183-L189
        self.intra_time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.emb_dim),
        )

        self.atom_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.model = args.model
        if self.model == "AssembleFlow_Atom":
            self.velocity_function = Velocity_Atom(
                hidden_dim=self.hidden_dim, hidden_coff_dim=128, time_embed_dim=self.time_embed_dim,
                num_layers=args.num_layers, num_convs=args.num_convs, num_head=args.num_head,
                dropout=0.1, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)
            self.get_output = self.get_output_02
        elif self.model == "AssembleFlow_Molecule":
            from easydict import EasyDict
            ipa_config = EasyDict({
                "num_blocks": 4,
                "s_dim": self.hidden_dim,
                "z_dim": self.hidden_dim,
                "c_hidden": self.hidden_dim,
                "qk_dim": 8,
                "v_dim": 12,
                "num_heads": 4,
            })
            self.velocity_function = EquivariantScoreNetwork_03(
                hidden_dim=self.hidden_dim,
                ipa_config=ipa_config, flow_matcher=None
            )
            self.get_output = self.get_output_03
        return

    def get_edge_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def get_output_01(self, x, sampled_timestep, position_t, atom2molecule, atom2cluster):
        edge_index = radius_graph(position_t, r=self.cutoff, batch=atom2molecule)
        device = x.device
        row, col = edge_index
        
        time_attr = self.time_embed(timestep_embedding(sampled_timestep.squeeze(1), self.emb_dim))  # [N, d]

        if self.model_3d == "PaiNN":
            radius_edge_index = edge_index
            _, atom_3d_repr = self.intra_model.forward_with_expanded_index(x, position_t, radius_edge_index, atom2molecule, return_node_repr=True)

        # compute representations and positions for mass center
        dim_size = atom2molecule.max().item() + 1

        molecule_repr = scatter(atom_3d_repr, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        molecule_repr = molecule_repr + self.intra_time_emb_layers(time_attr)
        molecule_position = scatter(position_t, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        molecule2cluster = torch.arange(dim_size).to(device) // 17
        
        edge_index = radius_graph(molecule_position, r=self.cluster_cutoff, batch=molecule2cluster)
        index_i, index_j = edge_index

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_intra=molecule_position, index_intra=index_j, pos_center=molecule_position, index_center=index_i)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        
        r_i, r_j = molecule_position[index_i], molecule_position[index_j]  # [num_edge, 3]
        coff_i = torch.einsum("abc,ac->ab", edge_basis, r_i)  # [num_edge, 3]
        coff_j = torch.einsum("abc,ac->ab", edge_basis, r_j)  # [num_edge, 3]
        embed_i = self.get_edge_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_edge_embedding(coff_j)  # [num_edge, C]

        edge_embed = torch.cat([embed_i, embed_j], dim=-1)
        edge_attr = self.project(edge_embed)

        velocity_translation, velocity_quaternion = self.velocity_function(
            graph_attr=molecule_repr, edge_index=edge_index, edge_attr=edge_attr, time_attr=time_attr, equivariant_basis=equivariant_basis,
        )
        return velocity_translation, velocity_quaternion

    def get_output_02(self, x, sampled_timestep, position_t, atom2molecule, atom2cluster):
        edge_index = radius_graph(position_t, r=self.cutoff, batch=atom2molecule)
        device = x.device
        row, col = edge_index
        
        time_attr = self.time_embed(timestep_embedding(sampled_timestep.squeeze(1), self.emb_dim))  # [N, d]

        if self.model_3d == "PaiNN":
            radius_edge_index = edge_index
            _, atom_3d_repr = self.intra_model.forward_with_expanded_index(x, position_t, radius_edge_index, atom2molecule, return_node_repr=True)

        atom_3d_repr = atom_3d_repr + self.intra_time_emb_layers(time_attr)

        # match dimension
        atom_repr = self.atom_emb(atom_3d_repr)

        # compute representations and positions for mass center
        dim_size = atom2molecule.max().item() + 1

        molecule_repr = scatter(atom_repr, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        position_molecule = scatter(position_t, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        molecule2cluster = torch.arange(dim_size).to(device) // 17
        
        edge_index = radius(position_t, position_molecule, self.cluster_cutoff, atom2cluster, molecule2cluster, max_num_neighbors=position_t.shape[0])
        index_molecule, index_atom = edge_index

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_intra=position_t, index_intra=index_atom, pos_center=position_molecule, index_center=index_molecule)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        
        r_i, r_j = position_t[index_atom], position_molecule[index_molecule]  # [num_edge, 3]
        coff_i = torch.einsum("abc,ac->ab", edge_basis, r_i)  # [num_edge, 3]
        coff_j = torch.einsum("abc,ac->ab", edge_basis, r_j)  # [num_edge, 3]
        embed_i = self.get_edge_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_edge_embedding(coff_j)  # [num_edge, C]

        edge_embed = torch.cat([embed_i, embed_j], dim=-1)
        edge_attr = self.project(edge_embed)

        velocity_translation, velocity_quaternion = self.velocity_function(
            node_attr=atom_repr, graph_attr=molecule_repr,
            edge_index=edge_index, edge_attr=edge_attr, time_attr=time_attr, equivariant_basis=equivariant_basis,
            intra_node2graph=atom2molecule)
        return velocity_translation, velocity_quaternion

    def get_output_03(self, x, sampled_timestep, translation_t, quaternion_t, position_t, atom2molecule, atom2cluster):
        edge_index = radius_graph(position_t, r=self.cutoff, batch=atom2molecule)
        device = x.device
        row, col = edge_index
        
        time_attr = self.time_embed(timestep_embedding(sampled_timestep.squeeze(1), self.emb_dim))  # [N, d]

        if self.model_3d == "PaiNN":
            radius_edge_index = edge_index
            _, atom_3d_repr = self.intra_model.forward_with_expanded_index(x, position_t, radius_edge_index, atom2molecule, return_node_repr=True)

        # compute representations and positions for mass center
        dim_size = atom2molecule.max().item() + 1

        molecule_repr = scatter(atom_3d_repr, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        molecule_repr = molecule_repr + self.intra_time_emb_layers(time_attr)
        molecule_position = scatter(position_t, atom2molecule, dim=0, dim_size=dim_size, reduce="mean")
        molecule2cluster = torch.arange(dim_size).to(device) // 17
        
        edge_index = radius_graph(molecule_position, r=self.cluster_cutoff, batch=molecule2cluster)
        index_i, index_j = edge_index

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_intra=molecule_position, index_intra=index_j, pos_center=molecule_position, index_center=index_i)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        
        r_i, r_j = molecule_position[index_i], molecule_position[index_j]  # [num_edge, 3]
        coff_i = torch.einsum("abc,ac->ab", edge_basis, r_i)  # [num_edge, 3]
        coff_j = torch.einsum("abc,ac->ab", edge_basis, r_j)  # [num_edge, 3]
        embed_i = self.get_edge_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_edge_embedding(coff_j)  # [num_edge, C]

        edge_embed = torch.cat([embed_i, embed_j], dim=-1)
        edge_attr = self.project(edge_embed)

        velocity_translation, velocity_quaternion = self.velocity_function(
            node_attr=molecule_repr,
            edge_index=edge_index, edge_attr=edge_attr, time_attr=time_attr, equivariant_basis=equivariant_basis,
            input_feats=(translation_t, quaternion_t),
            node2graph=atom2molecule,
        )
        return velocity_translation, velocity_quaternion

    def forward(self, data):
        X_0 = data.initial_positions
        X_1 = data.final_positions
        device = X_0.device

        atom2molecule, atom2cluster = data.intra_batch, data.batch
        """
            atom2molecule: [0, 1, ..., 16, 17, ..., 33]
            atom2cluster: [0, 0, ...,  0,  1, ...,  1]
        """
        num_cluster = data.num_graphs
        cluster_batch = torch.LongTensor([[i for _ in range(17)] for i in range(num_cluster)]).to(device).view(-1)

        # extract molecule-level rotation and translation
        rotation_0, translation_0 = data.initial_rotation, data.initial_translation  # [num_cluster*17, 3, 3], [num_cluster*17, 3]
        rotation_1, translation_1 = data.final_rotation, data.final_translation  # [num_cluster*17, 3, 3], [num_cluster*17, 3]
        # already normalized
        quaternion_0 = rot_to_quat(rotation_0)  # [num_cluster*17, 4]
        quaternion_1 = rot_to_quat(rotation_1)  # [num_cluster*17, 4]
        # # Normalize
        # normalized_quaternion_0 = quaternion_0 / torch.linalg.norm(quaternion_0, dim=-1, keepdim=True)
        # normalized_quaternion_1 = quaternion_1 / torch.linalg.norm(quaternion_1, dim=-1, keepdim=True)

        # sample timesteps
        sampled_timestep = torch.randint(0, self.num_timesteps, size=(num_cluster,), device=device)  # [num_cluster]
        # normalize to [0, 1]
        sampled_timestep = sampled_timestep / self.num_timesteps * (1 - EPSILON) + EPSILON  # [num_cluster]

        # TODO: may not require_grad
        sampled_timestep = sampled_timestep.requires_grad_(True)  # [num_cluster]
        translation_0 = translation_0.requires_grad_(True)
        translation_1 = translation_1.requires_grad_(True)
        quaternion_0 = quaternion_0.requires_grad_(True)
        quaternion_1 = quaternion_1.requires_grad_(True)

        # from graph-level to atom-level timesteps
        sampled_timestep_molecule = sampled_timestep.index_select(0, cluster_batch)  # [num_cluster*17]
        sampled_timestep_molecule = sampled_timestep_molecule.unsqueeze(1)  # [num_cluster*17, 1]
        translation_t = LERP(translation_0, translation_1, sampled_timestep_molecule)  # [num_cluster*17, 3]
        quaternion_t = SLERP(quaternion_0, quaternion_1, sampled_timestep_molecule)  # [num_cluster*17, 4]

        # from cluster-level to atom-level timesteps
        sampled_timestep_atom = sampled_timestep.index_select(0, atom2cluster).unsqueeze(1)  # [num_atom, 1]

        position_t = move_molecule(X_0, translation_0=translation_0, translation_t=translation_t, quaternion_0=quaternion_0, quaternion_t=quaternion_t, atom2molecule=atom2molecule) # [num_atom, 3]

        if self.model == "AssembleFlow_Atom":
            hat_translation_1, hat_quaternion_1 = self.get_output(data.x, sampled_timestep_atom, position_t, atom2molecule, atom2cluster) # [num_cluster*17, 3], [num_cluster*17, 4]
        elif self.model == "AssembleFlow_Molecule":
            hat_translation_1, hat_quaternion_1 = self.get_output(data.x, sampled_timestep_molecule, translation_t, quaternion_t, position_t, atom2molecule, atom2cluster) # [num_cluster*17, 3], [num_cluster*17, 4]
        
        hat_quaternion_1 = hat_quaternion_1 / (torch.linalg.norm(hat_quaternion_1, dim=-1, keepdim=True) + EPSILON)

        loss_translation = torch.mean((translation_1 - hat_translation_1) ** 2, -1)  # [num_cluster*17]
        loss_rotation = torch.mean((quaternion_1 - hat_quaternion_1) ** 2, -1)  # [num_cluster*17]

        loss = self.alpha_translation * loss_translation.mean() + self.alpha_rotation * loss_rotation.mean()

        return loss

    @torch.no_grad()
    def position_inference(self, data, inference_interval=1, verbose=False, step_size=1):
        translation_0 = translation = data.initial_translation
        rotation_0 = data.initial_rotation
        quaternion_0 = quaternion = rot_to_quat(rotation_0)
        X_0 = pos = data.initial_positions

        translation_1 = data.final_translation
        rotation_1 = data.final_rotation
        quaternion_1 = rot_to_quat(rotation_1)

        device = pos.device
        num_cluster = data.num_graphs
        num_atom = X_0.shape[0]

        timesteps = torch.arange(self.num_timesteps, device=device)
        timesteps = timesteps / self.num_timesteps * (1 - EPSILON) + EPSILON
        delta_t = 1. / self.num_timesteps

        interval_list, pos_list, translation_list, quaternion_list = [], [], [], []
        interval_list.append(0)
        pos_list.append(pos)
        translation_list.append(translation)
        quaternion_list.append(quaternion)

        if verbose:
            L = tqdm(range(1, 1+self.num_timesteps))
        else:
            L = range(1, 1+self.num_timesteps)
        
        for i in L:
            t = timesteps[i-1]

            if self.model == "AssembleFlow_Atom":
                sampled_timestep = torch.ones(num_atom).unsqueeze(1).to(device) * t
                pred_translation_1, pred_quaternion_1 = self.get_output(data.x, sampled_timestep, pos, data.intra_batch, data.batch)

            elif self.model == "AssembleFlow_Molecule":
                sampled_timestep = torch.ones(num_cluster * 17).unsqueeze(1).to(device) * t
                pred_translation_1, pred_quaternion_1 = self.get_output(data.x, sampled_timestep, translation, quaternion, pos, data.intra_batch, data.batch)

            pred_quaternion_1 = pred_quaternion_1 / (torch.linalg.norm(pred_quaternion_1, dim=-1, keepdim=True) + EPSILON)

            translation = translation + step_size * delta_t * (pred_translation_1 - translation) / (1-t)
            quaternion = quaternion + step_size * delta_t * SLERP_derivative(quaternion_0, pred_quaternion_1, t)
            quaternion = quaternion / (torch.linalg.norm(quaternion, dim=-1, keepdim=True) + EPSILON)

            pos = move_molecule(X_0, translation_0=translation_0, translation_t=translation, quaternion_0=quaternion_0, quaternion_t=quaternion, atom2molecule=data.intra_batch) # [num_atom, 3]

            if i % inference_interval == 0:
                interval_list.append(i)
                pos_list.append(pos)
                translation_list.append(translation)
                quaternion_list.append(quaternion)
        
        return interval_list, pos_list, translation_list, quaternion_list
