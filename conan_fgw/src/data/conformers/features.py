import copy
import logging
import random
from abc import ABC
from typing import List

import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import Conformer, AllChem, rdMolAlign
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from conan_fgw.src.data.conformers.generators import MolWithRepeatingConformers

python_logger = logging.getLogger("features")


class MolGraphFeaturizer(ABC):
    pass


class MolGraphFeaturizer2D(MolGraphFeaturizer):
    @staticmethod
    def featurize(smiles_list: List[str], y_list: List[float]) -> List:
        return [
            MolGraphFeaturizer2D._create_data_record(smiles_list[i], y_list[i])
            for i in range(len(smiles_list))
        ]

    @staticmethod
    def _create_data_record(mol_smiles: str, y: float) -> Data:
        data = torch_geometric.utils.smiles.from_smiles(mol_smiles)
        data.y = torch.tensor(y, dtype=torch.float)
        return data


class MolGraphFeaturizer3D(MolGraphFeaturizer):
    @staticmethod
    def featurize(
        mol_list: List[MolWithRepeatingConformers], y_list: List[float], num_conformers: int
    ):
        data_list, conformers_list = [], []
        for idx, mol_with_confs in enumerate(mol_list):
            conformers = mol_with_confs.get_conformers()
            # random.seed(1)
            # logging.info(f"🧨 Num of generated conformers: {len(conformers)} in MolGraphFeaturizer3D")
            if num_conformers > len(conformers):
                conformers = random.choices(conformers, k=num_conformers)
            elif num_conformers < len(conformers):
                conformers = random.sample(conformers, k=num_conformers)
            elif num_conformers == 0:
                raise ValueError(
                    f"num_conformers must be greater or equal than 1, but got {num_conformers}"
                )
            mol_data_list = [
                MolGraphFeaturizer3D._create_data_record(conf, mol_with_confs, y_list[idx])
                for conf in conformers
            ]
            data_list.append(Batch.from_data_list(mol_data_list))
            conformers_list.append(conformers)
        return data_list, conformers_list

    @staticmethod
    def featurize_conformers(
        mol_list: List[MolWithRepeatingConformers],
        conf_list: List[List[Conformer]],
        y_list: List[float],
    ):
        data_list, conformers_list = [], []
        for idx, conformers in enumerate(conf_list):
            mol_data_list = [
                MolGraphFeaturizer3D._create_data_record(conf, mol_list[idx], y_list[idx])
                for conf in conformers
            ]
            data_list.append(Batch.from_data_list(mol_data_list))
            conformers_list.append(conformers)
        return data_list, conformers_list

    @staticmethod
    def featurize_diverse(
        mol_list: List[MolWithRepeatingConformers],
        y_list: List[float],
        num_conformers: int,
        distance_matrix,
    ):
        data_list, conformers_list = [], []
        for idx, mol_with_confs in enumerate(mol_list):
            conformers = mol_with_confs.get_conformers()
            conformers = MolGraphFeaturizer3D.sample_diverse(
                conformers, distance_matrix, len(conformers), num_conformers
            )
            mol_data_list = [
                MolGraphFeaturizer3D._create_data_record(conf, mol_with_confs, y_list[idx])
                for conf in conformers
            ]
            data_list.append(Batch.from_data_list(mol_data_list))
            conformers_list.append(conformers)
        return data_list, conformers_list

    @staticmethod
    def featurize_diverse_clustering(
        mol_list: List[MolWithRepeatingConformers],
        y_list: List[float],
        num_conformers: int,
        distance_matrix,
    ):
        data_list, conformers_list = [], []
        kmed = KMedoids(n_clusters=num_conformers, metric="precomputed", random_state=42)  ## 0
        for idx, mol_with_confs in enumerate(mol_list):
            conformers = mol_with_confs.get_conformers()
            kmed.fit(distance_matrix)
            conformers = [conformers[t] for t in kmed.medoid_indices_]
            mol_data_list = [
                MolGraphFeaturizer3D._create_data_record(conf, mol_with_confs, y_list[idx])
                for conf in conformers
            ]
            data_list.append(Batch.from_data_list(mol_data_list))
            conformers_list.append(conformers)
        return data_list, conformers_list

    @staticmethod
    def populate_conformer(reference_data_point, pos):
        reference_data_point.pos = pos
        return reference_data_point

    @staticmethod
    def select_diverse_conformers(conformers, num_conformers_to_select):
        # Assume conformers is a list of RDKit conformer objects
        num_conformers = len(conformers)
        distance_matrix = [[0] * num_conformers for _ in range(num_conformers)]

        # Compute the distance matrix
        for i in tqdm(
            range(num_conformers), desc=f"Processing molecule f{conformers[0].GetOwningMol()}"
        ):
            for j in range(i + 1, num_conformers):
                distance = rdMolAlign.GetBestRMS(
                    conformers[i].GetOwningMol(), conformers[j].GetOwningMol(), i, j
                )
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        return MolGraphFeaturizer3D.sample_diverse(
            conformers, distance_matrix, num_conformers, num_conformers_to_select
        )

    @staticmethod
    def sample_diverse(conformers, distance_matrix, num_conformers, num_conformers_to_select):
        # Initialize the diversity set with the first conformer
        diverse_set = {random.randrange(num_conformers)}
        # Iterative selection
        while len(diverse_set) < num_conformers_to_select:
            max_dist = -1
            max_idx = -1
            for i in range(num_conformers):
                if i not in diverse_set:
                    min_dist_to_set = min(distance_matrix[i][j] for j in diverse_set)
                    if min_dist_to_set > max_dist:
                        max_dist = min_dist_to_set
                        max_idx = i
            diverse_set.add(max_idx)
        return [conformers[i] for i in diverse_set]

    @staticmethod
    def featurize_n_times(
        mol_list: List[MolWithRepeatingConformers],
        y_list: List[float],
        num_conformers: int,
        n_trials: int,
    ):
        data_list, conformers_list = [], []
        for idx, mol_with_confs in enumerate(mol_list):
            trail_data_list, trial_confs_list = [], []
            for i in range(n_trials):
                conformers = mol_with_confs.get_conformers()
                if num_conformers > len(conformers):
                    conformers = random.choices(conformers, k=num_conformers)
                elif num_conformers < len(conformers):
                    conformers = random.sample(conformers, k=num_conformers)
                elif num_conformers == 0:
                    raise ValueError(
                        f"num_conformers must be greater or equal than 1, but got {num_conformers}"
                    )
                mol_data_list = [
                    MolGraphFeaturizer3D._create_data_record(conf, mol_with_confs, y_list[idx])
                    for conf in conformers
                ]
                trail_data_list.append(Batch.from_data_list(mol_data_list))
                trial_confs_list.append(conformers)
            data_list.append(trail_data_list)
            conformers_list.append(trial_confs_list)
        return data_list, conformers_list

    @staticmethod
    def _create_data_record(
        conformer, mol_with_confs: MolWithRepeatingConformers, y: float
    ) -> Data:
        data = torch_geometric.utils.smiles.from_smiles(mol_with_confs.smiles, with_hydrogen=True)
        data.y = torch.tensor(y, dtype=torch.float)
        data.pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)
        data.z = torch.tensor(
            [atom.GetAtomicNum() for atom in mol_with_confs.mol.GetAtoms()], dtype=torch.long
        )
        return data

    @staticmethod
    def cantor_pairing_function(x, y):
        return (x + y) * (x + y + 1) / 2 + y
