import os
from tqdm import tqdm
import numpy as np
import random
from itertools import repeat, permutations
from collections import defaultdict
from pymatgen.core.structure import Structure

import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_scatter import scatter
import importlib.resources
import AssembleFlow

from .dataset_utils import PeriodicTable


def compute_Inertia_Tensor(positions):
    mass_center = torch.mean(positions, dim=0, keepdim=True)
    
    positions = positions - mass_center
    # I_each = np.dot(r, r) * torch.eye(3) - torch.outer(r, r)
    
    eye = torch.eye(3) # [3, 3]
    inner_product = torch.einsum("bi,bi->b", positions, positions) # [N]
    outer_product = torch.einsum("bi,bj->bij", positions, positions)  # [N, 3, 3]

    I_atom = inner_product.unsqueeze(1).unsqueeze(2) * eye.unsqueeze(0) - outer_product  # [N, 3, 3]

    I = torch.mean(I_atom, dim=0)
    return I


def ensure_right_handedness(eigenvectors):
    # 确保右手性
    # 检查右手性
    cross_product = torch.cross(eigenvectors[:, 0], eigenvectors[:, 1])
    if torch.dot(cross_product, eigenvectors[:, 2]) < 0:
        # 如果不符合右手性，则反转第三个特征向量
        eigenvectors[:, 2] = -eigenvectors[:, 2]
    return eigenvectors


def determine_handness(vectors):
    # vectors 是一个 3x3 矩阵，其中每一列是一个特征向量
    determinant = np.linalg.det(vectors)
    if determinant > 0:
        return "Right-handed"
    else:
        return "Left-handed"


def check_tie_index(eigen_values):
    EPS = 1e-3
    tie_index_set = set()
    if abs(eigen_values[0] - eigen_values[1]) <= EPS:
        tie_index_set.add(0)
        tie_index_set.add(1)
    if abs(eigen_values[1] - eigen_values[2]) <= EPS:
        tie_index_set.add(1)
        tie_index_set.add(2)
    tie_index_list = sorted(list(tie_index_set))
    return tie_index_list


def extract_rotation_matrix(positions):
    inertial_tensors = compute_Inertia_Tensor(positions)
    eigen_values, eigen_vectors = torch.linalg.eigh(inertial_tensors)  # [basis 0, basis 1, basis 2]
    rotation = eigen_vectors
    rotation = ensure_right_handedness(rotation)
    handness = determine_handness(rotation)
    assert handness == "Right-handed", "Rotation matrix should be right-hand."
    tie_index_list = check_tie_index(eigen_values)
    return rotation, tie_index_list


def check_rotation_directions(input_initial_rotation, input_final_rotation, initial_pos_normalized, final_pos_normalized):
    '''
    input_initial_rotation is guaranteed to be righhandness
    '''
    initial_rotation = input_initial_rotation.clone()
    final_rotation = input_final_rotation.clone()

    def get_non_trivial_anchor_points(use_perpendicular=True):
        # pickup a non-trivial atom pair
        # EPS = 1e-1
        EPS = 1e-3
        threshold = 5
        for i in range(initial_pos_normalized.shape[0]):
            valid = True
            for j in range(3):
                # initial_angle = torch.acos(torch.dot(initial_pos_normalized[i], initial_rotation[:, j])) * 180 / np.pi
                # final_angle = torch.acos(torch.dot(final_pos_normalized[i], final_rotation[:, j])) * 180 / np.pi

                # if -threshold <= initial_angle <= threshold:
                #     valid = False
                #     break
                # if -threshold <= final_angle <= threshold:
                #     valid = False
                #     break
                # if use_perpendicular and 90-threshold <= initial_angle <= 90+threshold:
                #     valid = False
                #     break
                # if use_perpendicular and 90-threshold <= final_angle <= 90+threshold:
                #     valid = False
                #     break

                # When theta-->0, cosine(theta)-->1
                if torch.abs(torch.dot(initial_pos_normalized[i], initial_rotation[:, j])).item() > 1-EPS:
                    valid = False
                    break
                if torch.abs(torch.dot(final_pos_normalized[i], final_rotation[:, j])).item() > 1-EPS:
                    valid = False
                    break
                if use_perpendicular:
                    # When theta-->90, cosine(theta)-->0, means the anchor points are perpendicular
                    if torch.abs(torch.dot(initial_pos_normalized[i], initial_rotation[:, j])).item() < EPS:
                        valid = False
                        break
                    if torch.abs(torch.dot(final_pos_normalized[i], final_rotation[:, j])).item() < EPS:
                        valid = False
                        break
            if valid:
                return initial_pos_normalized[i], final_pos_normalized[i]
        return None, None
                
    initial_pos_anchor, final_pos_anchor = get_non_trivial_anchor_points(use_perpendicular=True)
    if initial_pos_anchor is None:
        # print("Cannot find non-perpendicular anchors. Molecule is flat.", if_molecule_is_flat)
        # This means the molecule is flat.
        # assert if_molecule_is_flat, "We cannot find a non-perpendicular anchor atom, only when molecule is flat."
        initial_pos_anchor, final_pos_anchor = get_non_trivial_anchor_points(use_perpendicular=False)
    
    # start checking
    # EPS = 1e-1
    threshold = 5
    for i in range(3):
        # angle = torch.dot(initial_pos_anchor, initial_rotation[:, i]) * torch.dot(final_pos_anchor, final_rotation[:, i])
        angle_01 = torch.acos(torch.dot(initial_pos_anchor, initial_rotation[:, i])) * 180 / np.pi
        angle_02 = torch.acos(torch.dot(final_pos_anchor, final_rotation[:, i])) * 180 / np.pi
        # print("angle of {} is {}".format(i, angle), angle_01, angle_02)
        # if 180 - 2*threshold<= angle_01 + angle_02 <= 180 + 2*threshold:
        if torch.dot(initial_pos_anchor, initial_rotation[:, i]) * torch.dot(final_pos_anchor, final_rotation[:, i]) < 0:
            final_rotation[:, i] = -final_rotation[:, i]
            # print("change")

    ########## Check rotation matrix for reconstruction ##########
    """
    I^T R = F^T
    R = I F^T
    initial_pos R = final_pos
    """
    SO3_rotation_matrix = torch.matmul(initial_rotation, final_rotation.T)
    final_pos_prime = torch.matmul(initial_pos_normalized, SO3_rotation_matrix)
    residue = final_pos_normalized - final_pos_prime
    residue = (residue**2).mean()
    return residue.item(), initial_rotation, final_rotation


def extract_data_from_file(file_path, periodic_table):
    f = open(file_path, "r")

    data_record = defaultdict(list)

    f.readline()
    for line in f.readlines():
        line = line.strip()
        mol_idx, atom_type, x, y, z = line.split(",")
        mol_idx = int(mol_idx)
        atom_idx = periodic_table.get_atomic_index(atom_type)
        x = float(x)
        y = float(y)
        z = float(z)
        final_positions = np.array([x, y, z])

        data_record[mol_idx].append([atom_idx, final_positions])
    
    assert len(data_record) == 17
    num_atom_each_molecule = set([len(v) for k,v in data_record.items()])
    assert len(num_atom_each_molecule) == 1

    bulk_data_list = []
    for mol_idx, item_list in data_record.items():
        atom_feature_list, positions_list = [], []

        for item in item_list:
            atom_idx, positions = item
            atomic_number = atom_idx + 1

            atomic_mass = periodic_table.get_atomic_mass(atomic_number)
            atomic_radius = periodic_table.get_atomic_radius(atomic_number)
            electronegativity = periodic_table.get_electronegativity(atomic_number)
            ionization_energy = periodic_table.get_ionization_energy(atomic_number)
            oxidation_states = periodic_table.get_oxidation_states(atomic_number)

            # TODO: this will throw out the following exception
            # value cannot be converted to type int64 without overflow
            # atom_feature = [atom_idx, atomic_mass, atomic_radius, electronegativity, ionization_energy] + oxidation_states
            atom_feature = [atom_idx]
            atom_feature_list.append(atom_feature)
            positions_list.append(positions)

        atom_feature_list = torch.tensor(atom_feature_list, dtype=torch.int64)
        positions_list = np.array(positions_list)
        positions_list = torch.tensor(positions_list, dtype=torch.float32)
        
        data = dict(
            x=atom_feature_list,
            positions=positions_list,
        )
        bulk_data_list.append(data)
        
    return bulk_data_list


class CrystallizationDatasetCOD(InMemoryDataset):
    def __init__(self, root, subset=-1):
        self.root = root
        self.subset = subset
        super(CrystallizationDatasetCOD, self).__init__(root, None, None, None)

        self.saved_data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return None

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "valid_idx.txt", "processed_valid_idx.txt"]

    @property
    def processed_dir(self):
        if self.subset == -1:
            return os.path.join(self.root, "processed")
        else:
            return os.path.join(self.root, "processed_{}".format(self.subset))

    def get(self, idx):
        target_keys = [
            "x",
            "initial_positions", "initial_rotation", "initial_translation",
            "final_positions", "final_rotation", "final_translation",
            "intra_batch"
        ]

        bulk_data = Data()
        for key in target_keys:
            item, slices = self.saved_data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[bulk_data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            bulk_data[key] = item[s]
        
        return bulk_data

    def process(self):
        print("preprocessing...")
        try:
            periodic_table_file = importlib.resources.as_file(AssembleFlow.datasets, 'periodic_table.csv')
        except:
            with importlib.resources.path(AssembleFlow.datasets, 'periodic_table.csv') as file_name:
                periodic_table_file = file_name
        periodic_table = PeriodicTable(periodic_table_file)

        file_list = os.listdir(os.path.join(self.raw_dir, "FinalPositions"))
        file_list = filter(lambda x: x.endswith("csv"), file_list)
        file_idx_list = [x.replace("_cluster16.csv", "") for x in file_list]
        file_idx_list = sorted(file_idx_list)
        print("len file_list", len(file_idx_list)) # 106672

        if self.subset != -1:
            f = open(self.processed_paths[1], "r")
            file_idx_list = []
            for line in f.readlines():
                file_idx_list.append(line.strip())

        data_list = []
        valid_file_idx_list = []
        
        for idx, file_idx in enumerate(tqdm(file_idx_list)):
            # print()
            # print(file_idx)
            if_any_molecule_in_cluster_flat = False
            try: 
                initial_file_path = os.path.join(self.raw_dir, "InitialPositions", "{}_cluster16_noised.csv".format(file_idx))
                final_file_path = os.path.join(self.raw_dir, "FinalPositions", "{}_cluster16.csv".format(file_idx))

                initial_bulk_data_list = extract_data_from_file(initial_file_path, periodic_table)
                final_bulk_data_list = extract_data_from_file(final_file_path, periodic_table)

                bulk_data_list = []
                for initial_data, final_data in zip(initial_bulk_data_list, final_bulk_data_list):
                    assert torch.equal(initial_data["x"], final_data["x"])

                    x = initial_data["x"]
                    initial_positions = initial_data["positions"]
                    final_positions = final_data["positions"]

                    initial_rotation, initial_tie_list = extract_rotation_matrix(initial_positions)
                    final_rotation, final_tie_list = extract_rotation_matrix(final_positions)

                    initial_pos_centered = initial_positions - torch.mean(initial_positions, dim=0, keepdim=True)
                    initial_pos_norm = torch.norm(initial_pos_centered, dim=1, keepdim=True)
                    final_pos_centered = final_positions - torch.mean(final_positions, dim=0, keepdim=True)
                    final_pos_norm = torch.norm(final_pos_centered, dim=1, keepdim=True)

                    # normalize points that are very close to the origin
                    EPS = 1e-5
                    L = x.shape[0]
                    for i in range(L):
                        if initial_pos_norm[i] <= EPS:
                            assert final_pos_norm[i] <= EPS
                            initial_pos_norm[i] = EPS
                            final_pos_norm[i] = EPS
                            initial_pos_centered[i] = 0
                            final_pos_centered[i] = 0
                    initial_pos_normalized = initial_pos_centered / initial_pos_norm
                    final_pos_normalized = final_pos_centered / final_pos_norm

                    initial_rank = torch.linalg.matrix_rank(initial_pos_centered)
                    final_rank = torch.linalg.matrix_rank(final_pos_normalized)
                    if_molecule_is_flat = initial_rank < 3
                    if_any_molecule_in_cluster_flat = if_any_molecule_in_cluster_flat or if_molecule_is_flat
                    
                    assert initial_tie_list == final_tie_list, "Two ties lists should be the same. {}. {}.".format(initial_tie_list, final_tie_list)
                    order_list = [[0, 1, 2]]
                    if initial_tie_list == [0, 1]:
                        order_list.append([1, 0, 2])
                    elif initial_tie_list == [1, 2]:
                        order_list.append([0, 2, 1])
                    elif initial_tie_list == [0, 1, 2]:
                        order_list = list(permutations(order_list[0]))
                        order_list = [list(x) for x in order_list]
                    # print(initial_tie_list)
                    optimal_residue, optimal_order, optimal_initial_rotation, optimal_final_rotation = 1e10, [0, 1, 2], initial_rotation.clone(), final_rotation.clone()
                    
                    for order in order_list:
                        residue, neo_initial_rotation, neo_final_rotation = check_rotation_directions(initial_rotation, final_rotation[:, order], initial_pos_normalized, final_pos_normalized)
                        if residue < optimal_residue:
                            optimal_residue = residue
                            optimal_order = order
                            optimal_initial_rotation = neo_initial_rotation.clone()
                            optimal_final_rotation = neo_final_rotation.clone()

                    EPS = 1e-3
                    assert optimal_residue <= EPS, "Optimal residue {} should be smaller than {}.".format(optimal_residue, EPS)
                    initial_rotation = optimal_initial_rotation.clone()
                    final_rotation = optimal_final_rotation.clone()

                    data = Data(
                        x=x,
                        initial_positions=initial_positions,
                        final_positions=final_positions,
                        initial_rotation=initial_rotation.unsqueeze(0),
                        final_rotation=final_rotation.unsqueeze(0),
                    )
                    bulk_data_list.append(data)

                bulk_data_list = Batch.from_data_list(bulk_data_list)
                initial_positions_mass_center = bulk_data_list.initial_positions.mean(dim=0, keepdim=True)
                bulk_data_list.initial_positions = bulk_data_list.initial_positions - initial_positions_mass_center
                bulk_data_list.final_positions = bulk_data_list.final_positions - initial_positions_mass_center

                intra_node2graph = bulk_data_list.batch
                num_molecule = intra_node2graph.max().item() + 1
                initial_translation = scatter(bulk_data_list.initial_positions, intra_node2graph, dim=0, dim_size=num_molecule, reduce="mean")  # [num_graph, 1]
                final_translation = scatter(bulk_data_list.final_positions, intra_node2graph, dim=0, dim_size=num_molecule, reduce="mean")  # [num_graph, 1]

                assert not if_any_molecule_in_cluster_flat, "This cluster has flat molecules. Skip."

                bulk_data = Data(
                    x=bulk_data_list.x,
                    initial_positions=bulk_data_list.initial_positions,
                    initial_rotation=bulk_data_list.initial_rotation,
                    initial_translation=initial_translation,
                    final_positions=bulk_data_list.final_positions,
                    final_rotation=bulk_data_list.final_rotation,
                    final_translation=final_translation,
                    intra_batch=bulk_data_list.batch,
                )

                data_list.append(bulk_data)
                valid_file_idx_list.append(file_idx)

            except Exception as error:
                print(initial_file_path, final_file_path)
                print(error)
                continue

        print("{} valid".format(len(data_list)))  # 106423
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        f_ = open(self.processed_paths[2], "w")
        for idx in valid_file_idx_list:
            print(idx, file=f_)
        f_.close()

        return
