from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv


class GATLayer(nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.2):
        super(GATLayer, self).__init__()

        assert hidden_dim % n_head == 0
        self.MHA = TransformerConv(
            in_channels=hidden_dim,
            out_channels=int(hidden_dim // n_head),
            heads=n_head,
            dropout=dropout,
            edge_dim=hidden_dim,
        )
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, node_attr_intra, node_attr_center, edge_attr):
        neo_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x = (node_attr_center, node_attr_intra)
        x = self.MHA(x, edge_index, edge_attr)
        node_attr_intra = node_attr_intra + self.norm1(x)
        x = self.FFN(node_attr_intra)
        node_attr_intra = node_attr_intra + self.norm2(x)
        return node_attr_intra


class EquiLayer(MessagePassing):
    def __init__(self, eps=0., train_eps=False, activation="silu", **kwargs):
        super(EquiLayer, self).__init__(aggr='mean', **kwargs)
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None   

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, node_attr_intra, node_attr_center, edge_index, edge_attr, size=None):
        x: OptPairTensor = (node_attr_center, node_attr_intra)
        
        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor: 
        if self.activation:
            return self.activation(x_j + edge_attr)
        else: # TODO: we are mostly using False for activation
            return edge_attr


class Velocity_Atom(torch.nn.Module):
    def __init__(self, hidden_dim, hidden_coff_dim, time_embed_dim, num_layers, num_convs, num_head, dropout=0.1, activation="silu", short_cut=False, concat_hidden=False):
        super(Velocity_Atom, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.num_head = num_head
        self.dropout = dropout
        self.concat_hidden = concat_hidden
        self.hidden_coff_dim = hidden_coff_dim
        self.time_embed_dim = time_embed_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.gnn_layers = nn.ModuleList()
        self.R3_equi_modules = nn.ModuleList()
        self.SO3_equi_modules = nn.ModuleList()
        self.R3_basis_mlp_modules = nn.ModuleList()
        self.SO3_basis_mlp_modules = nn.ModuleList()
        self.SO3_project_modeuls = nn.ModuleList()
        self.time_embed_modules = nn.ModuleList()

        for _ in range(self.num_layers):

            trans_convs = nn.ModuleList()
            for _ in range(self.num_convs):
                trans_convs.append(GATLayer(self.num_head, self.hidden_dim, dropout=self.dropout))
            self.gnn_layers.append(trans_convs)

            self.time_embed_modules.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.hidden_dim),
            ))

            self.R3_basis_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                # nn.Softplus(),
                nn.SiLU(),
                nn.Linear(self.hidden_coff_dim, 3))
            )

            self.SO3_basis_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                # nn.Softplus(),
                nn.SiLU(),
                nn.Linear(self.hidden_coff_dim, 3))
            )
            self.SO3_project_modeuls.append(nn.Linear(3, 4))

            self.R3_equi_modules.append(EquiLayer(activation=False))
            self.SO3_equi_modules.append(EquiLayer(activation=False))

    def forward(self, node_attr, graph_attr, edge_index, edge_attr, time_attr, equivariant_basis, intra_node2graph):
        """
        Args:
            edge_index: edge connection (num_edge, 2)
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
            time_attr: time feature tensor with shape (num_node, hidden_time)
            equivariant_basis: an equivariant basis coord_diff, coord_cross, coord_vertical
        Output:
            velocity_translation
            velocity_rotation
        """
        hiddens = []
        dim_size = intra_node2graph.max().item() + 1
        coord_diff, coord_cross, coord_vertical = equivariant_basis
        index_center, index_intra = edge_index

        R3_velocity, SO3_velocity = 0, 0

        for layer_idx, (gnn_layer, time_embed, R3_basis_mlp_module, SO3_basis_mlp_module, R3_equi_module, SO3_equi_module) in enumerate(zip(self.gnn_layers, self.time_embed_modules, self.R3_basis_mlp_modules, self.SO3_basis_mlp_modules, self.R3_equi_modules, self.SO3_equi_modules)):

            for conv_idx, gnn in enumerate(gnn_layer):
                hidden = gnn(edge_index, node_attr, graph_attr, edge_attr)

                hidden = hidden + time_embed(time_attr)

                if conv_idx < len(gnn_layer) - 1 and self.activation is not None:
                    hidden = self.activation(hidden)
                assert hidden.shape == node_attr.shape
                if self.short_cut and hidden.shape == node_attr.shape:
                    hidden = hidden + node_attr

                hiddens.append(hidden)

            node_attr = hidden
            graph_attr = scatter(node_attr, intra_node2graph, dim=0, dim_size=dim_size, reduce="mean")

            h_row = node_attr[index_intra]
            h_col = graph_attr[index_center]

            edge_feature = torch.cat([h_row + h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)

            # generate velocity for R3
            R3_dynamic_coff = R3_basis_mlp_module(edge_feature)  # (num_edge, 3)
            R3_basis_mix = R3_dynamic_coff[:, :1] * coord_diff + R3_dynamic_coff[:, 1:2] * coord_cross + R3_dynamic_coff[:, 2:3] * coord_vertical  # (num_edge, 3)
            R3_velocity = R3_velocity + scatter(R3_equi_module(node_attr, graph_attr, edge_index, R3_basis_mix), intra_node2graph, dim=0, reduce="mean")  # (num_molecule, 3)

            # generate velocity for SO3
            SO3_dynamic_coff = SO3_basis_mlp_module(edge_feature)  # (num_edge, 3)
            SO3_basis_mix = SO3_dynamic_coff[:, :1] * coord_diff + SO3_dynamic_coff[:, 1:2] * coord_cross + SO3_dynamic_coff[:, 2:3] * coord_vertical  # (num_edge, 3)

            SO3_basis_mix = self.SO3_project_modeuls[layer_idx](SO3_basis_mix)  # (num_edge, 4)
            SO3_basis_mix = SO3_basis_mix / torch.linalg.norm(SO3_basis_mix, dim=-1, keepdim=True)  # (num_edge, 4)
            
            SO3_velocity = SO3_velocity + scatter(SO3_equi_module(node_attr, graph_attr, edge_index, SO3_basis_mix), intra_node2graph, dim=0, reduce="mean")  # (num_molecule, 3)

        return R3_velocity, SO3_velocity
