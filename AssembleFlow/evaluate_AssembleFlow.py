import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from itertools import repeat

import torch
from torch_geometric.data import Data
from torch_scatter import scatter


def calculate_packing_matching(true_positions, pred_positions):
    """
    https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/packing_similarity.html

    :true_positions: [N, 3]
    :pred_positions: [N, 3]
    """

    # Calculate pairwise distances between atoms in each structure
    true_distance = true_positions.unsqueeze(2) - true_positions.transpose(0, 1).unsqueeze(0)  # [N, 3, N]
    true_distance = torch.norm(true_distance, dim=1)  # [N, N]
    pred_distance = pred_positions.unsqueeze(2) - pred_positions.transpose(0, 1).unsqueeze(0)  # [N, 3, N]
    pred_distance = torch.norm(pred_distance, dim=1)  # [N, N]
    
    # Compute packing similarity metric (e.g., RMSD between distance matrices)
    similarity = torch.sqrt(torch.mean((true_distance - pred_distance) ** 2))

    return similarity.item()


# https://en.wikipedia.org/wiki/Covalent_radius#:~:text=The%20covalent%20radius%2C%20rcov,)%20%2B%20r(B).
covalent_radii_dict = {
    1:  0.31,
    5:  0.84,
    6:  0.69,
    7:  0.71,
    8:  0.66,
    9:  0.57,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    90: 2.06,
    91: 2.00,
    92: 1.96,
    93: 1.9,
    94: 1.87,
}


def evaluate_crystallization(file_path, num_timesteps, return_data_list=False):
    pred_key = "pred_positions_tid_{}".format(num_timesteps)
    keys = ["x", "initial_positions", "final_positions", "intra_batch", "repeat_idx_to_idx", pred_key]

    saved_data, saved_slices = torch.load(file_path)
    total_num_graph = saved_data.repeat_idx_to_idx.shape[0]
    data_list = []
    for idx in range(total_num_graph):
        data = Data()
        for key in keys:
            if saved_slices != None:
                item, slices = saved_data[key], saved_slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
            else:
                data = saved_data
        data_list.append([data, data.repeat_idx_to_idx.item()])
    
    packing_matching_atom_wise_dict, packing_matching_mass_center_dict = defaultdict(list), defaultdict(list)
    collision_dict = defaultdict(list)
    separation_dict = defaultdict(list)
    for data_idx, data_item in enumerate(data_list):
        data = data_item[0]
        graph_id = data_item[1]
        true_final_positions = data["final_positions"]
        pred_final_positions = data[pred_key]
        packing_matching_atom_wise = calculate_packing_matching(true_final_positions, pred_final_positions)
        packing_matching_atom_wise_dict[graph_id].append(packing_matching_atom_wise)


        intra_batch = data["intra_batch"]
        final_positions_mass_center = scatter(true_final_positions, intra_batch, dim=0, reduce="mean")
        pred_positions_mass_center = scatter(pred_final_positions, intra_batch, dim=0, reduce="mean")
        packing_matching_mass_center = calculate_packing_matching(final_positions_mass_center, pred_positions_mass_center)
        packing_matching_mass_center_dict[graph_id].append(packing_matching_mass_center)


        pred_distance = pred_final_positions.unsqueeze(2) - pred_final_positions.transpose(0, 1).unsqueeze(0)  # [N, 3, N]
        pred_distance = torch.norm(pred_distance, dim=1)  # [N, N]
        covalent_radii_list = []
        atom_type = data["x"]
        for x in atom_type:
            covalent_radii_list.append(covalent_radii_dict[x.item()+1])
        covalent_radii_list = torch.FloatTensor(covalent_radii_list)
        covalent_bond_distance = covalent_radii_list.unsqueeze(0) + covalent_radii_list.unsqueeze(1)
        covalent_bond_distance = covalent_bond_distance.to(pred_distance.device)
        N = pred_distance.shape[0]        
        total_count = N * N
        pred_collision_count = torch.sum(covalent_bond_distance > pred_distance)
        collision_dict[graph_id].append(100. * pred_collision_count.item() / total_count)


        separation_count = 0
        total_count = 0
        n = N // 17
        assert n * 17 == N
        for i in range(17):
            for j in range(i+1, 17):
                total_count += 1
                current_pred_distance = pred_distance[i*n: (i+1)*n, j*n: (j+1) * n]
                if torch.min(current_pred_distance) > 0.5:
                    separation_count += 1
        separation_dict[graph_id].append(100. * separation_count / total_count)


    packing_matching_atom_wise_list = []
    for k,v in packing_matching_atom_wise_dict.items():
        packing_matching_atom_wise_list.append(np.min(v))
    packing_matching_mass_center_list = []
    for k,v in packing_matching_mass_center_dict.items():
        packing_matching_mass_center_list.append(np.min(v))
    collision_list = []
    for k,v in collision_dict.items():
        collision_list.append(np.min(v))
    separation_list = []
    for k, v in separation_dict.items():
        separation_list.append(np.min(v))

    if return_data_list:
        return packing_matching_atom_wise_list, packing_matching_mass_center_list, collision_list, separation_list, data_list
    else:
        return packing_matching_atom_wise_list, packing_matching_mass_center_list, collision_list, separation_list
