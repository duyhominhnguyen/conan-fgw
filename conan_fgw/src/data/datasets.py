import json
import logging
import os
import pickle
from argparse import Namespace
from io import StringIO
from typing import List

import pandas as pd
import torch
from rdkit.Chem import Conformer
from torch_geometric.data import Dataset, Batch

from conan_fgw.src.data.conformers.features import (
    MolGraphFeaturizer3D,
    MolGraphFeaturizer2D,
)
from conan_fgw.src.data.conformers.generators import (
    RDKitConformersGenerator,
    MolWithRepeatingConformers,
)

from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.ipython_3d = True

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
import rdkit
import py3Dmol

python_logger = logging.getLogger("datasets")


def drawit(
    m,
    cids=[-1],
    p=None,
    colors=(
        "cyanCarbon",
        "redCarbon",
        "blueCarbon",
        "magentaCarbon",
        "whiteCarbon",
        "purpleCarbon",
    ),
):
    if p is None:
        p = py3Dmol.view(width=400, height=400)
    p.removeAllModels()
    for i, cid in enumerate(cids):
        IPythonConsole.addMolToView(m, p, confId=cid)
    for i, cid in enumerate(cids):
        p.setStyle(
            {
                "model": i,
            },
            {"stick": {"colorscheme": colors[i % len(colors)]}},
        )
    p.zoomTo()
    # return p.show()
    return p


class SmilesBasedDataset(Dataset):

    def __init__(self, mode: str, data_dir: str, config: Namespace, dataset_idx: int):
        super().__init__(None, None, None)
        self.data_dir = data_dir
        self.dataset_name = config.dataset_name[dataset_idx]
        self.featurizer = MolGraphFeaturizer2D()
        df = pd.read_csv(f"{data_dir}/{config.dataset_name[dataset_idx]}/{mode}.csv")
        smiles_list = df["smiles"].to_numpy()
        y_list = df[config.target[dataset_idx]].to_numpy()
        self.data_list = self.featurizer.featurize(smiles_list, y_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def calculate_len(path_to_file):
    num_lines = -1  # the first row is a header
    with open(path_to_file, "r") as file:
        for line in file:
            if line.strip() != "":
                num_lines += 1
    return num_lines


def load_conformer(conformers_dir, mol_id) -> List[MolWithRepeatingConformers]:
    mol_id = RDKitConformersGenerator.conformers_filename_prefix(mol_id.strip())
    file_name = f"{mol_id}.pkl"
    files = os.listdir(conformers_dir)
    if file_name not in files:
        raise ValueError(f"Conformers for molecule {mol_id} not found")
    file_path = os.path.join(conformers_dir, file_name)
    with open(file_path, "rb") as f:
        mol = pickle.load(f)
    return [mol]


class LargeConformerBasedDataset(Dataset):
    def __init__(self, mode: str, data_dir: str, config: Namespace, dataset_idx: int, fold=None):
        super().__init__()
        self.mode = mode
        self.dataset_fields = ["smiles", config.target[dataset_idx], "mol_id"]
        self.conformers_generator = RDKitConformersGenerator()
        self.featurizer = MolGraphFeaturizer3D()
        self.dataset_name = config.dataset_name[dataset_idx]
        if fold:
            self.data_dir = data_dir
            self.conformers_dir = f"{self.data_dir}/{self.dataset_name}/{fold}/conformers_{mode}"
            self.data_file_path = f"{self.data_dir}/{self.dataset_name}/{fold}/{mode}.csv"
        else:
            self.data_dir = data_dir
            self.conformers_dir = f"{self.data_dir}/{self.dataset_name}/conformers_{mode}"
            self.data_file_path = f"{self.data_dir}/{self.dataset_name}/{mode}.csv"

        df_dset = pd.read_csv(self.data_file_path)
        self.groundtruth = df_dset[config.target[dataset_idx]].tolist()
        self.num_conformers = config.num_conformers

        self.n_samples = calculate_len(self.data_file_path)

    def len(self):
        return self.n_samples

    def get(self, idx):
        sample = self._get_element(idx)
        mol_id = sample[:, 2].astype(str)[0].strip()
        y_list = [sample[:, 1].astype(float)]
        assert os.path.exists(self.conformers_dir), "Conformers directory does not exist"
        mols = load_conformer(self.conformers_dir, mol_id)
        # logging.info('Featurizing conformer')
        data_list, conformers_list = self.featurizer.featurize(
            mols, y_list, self.num_conformers
        )  ## num_conformers is set in the config file
        # logging.info(f'Featurized {len(conformers)} conformer')

        # logging.info('Creating num_atoms_index')
        num_atoms_index = LargeConformerBasedDataset.create_num_atoms_index(conformers_list)
        # logging.info('Created num_atoms_index')
        return data_list[0], num_atoms_index[0]

    def _get_element(self, idx):
        with open(self.data_file_path, "r") as file:
            header = file.readline()
            for line_idx, line in enumerate(file):
                if idx == line_idx:
                    csv_string = f"{header}{line.strip()}"
                    df = pd.read_csv(StringIO(csv_string))[self.dataset_fields]
                    return df.to_numpy()

    @staticmethod
    def create_num_atoms_index(conformers: List[List[Conformer]]) -> List[List[int]]:
        node_index = []  # Batch indices assigning each atom to a separate molecule
        for mol_conformers in conformers:
            mol_nodes = []
            for mol_conformer in mol_conformers:
                m_nodes = mol_conformer.GetOwningMol().GetNumAtoms()
                mol_nodes.append(m_nodes)
            node_index.append(mol_nodes)
        return node_index

    @staticmethod
    def collate_fn(batch):
        """
        batch: (B, (return from get method))
        """
        mol_id = -1
        mol_batch_index = []
        conf_node_batch = []  ## add in confnet_dss
        node_count = 0  ## add in confnet_dss

        for i in range(len(batch)):
            """
            Ref: MoleculeX/BasicProp/kddcup2021/conformer/dataset.py
            """
            mol_encodings = batch[i][1]  ## confs
            for j in range(len(mol_encodings)):
                mol_id += 1
                mol_batch_index.append(torch.Tensor([mol_id for _ in range(mol_encodings[j])]))

            num_confs = len(mol_encodings)
            num_nodes = mol_encodings[0]
            conf_node_batch.extend((torch.arange(num_nodes) + node_count).repeat(num_confs))
            node_count += num_nodes

        data_batch = Batch.from_data_list(
            [item for sublist in [t[0].to_data_list() for t in batch] for item in sublist]
        )
        data_batch.conf_node_batch = torch.LongTensor(conf_node_batch)
        batch_node_index = torch.cat(tuple(mol_batch_index)).long()
        return data_batch, batch_node_index

    @staticmethod
    def collate_fn_visual(batch):
        """
        batch: (B, (return from get method))
        """
        mol_id = -1
        mol_batch_index = []
        ## keep conformers
        conformers = []
        for i in range(len(batch)):
            mol_encodings = batch[i][1]
            conformers.append(batch[i][2])
            for j in range(len(mol_encodings)):
                mol_id += 1
                mol_batch_index.append(torch.Tensor([mol_id for _ in range(mol_encodings[j])]))
        data_batch = Batch.from_data_list(
            [item for sublist in [t[0].to_data_list() for t in batch] for item in sublist]
        )
        batch_node_index = torch.cat(tuple(mol_batch_index)).long()
        return data_batch, batch_node_index, conformers


class BDEDataset(LargeConformerBasedDataset):
    def __init__(self, mode: str, data_dir: str, config: Namespace, dataset_idx: int, fold=None):
        super().__init__(mode, data_dir, config, dataset_idx, fold)
        self.featurizer = MolGraphFeaturizerBDE()
        self.data_dir = data_dir
        self.dataset_fields = ["smiles", config.target[dataset_idx], "mol_id"]

    def _load_conformer(self, smiles, conformers_dir, mol_id) -> List[MolWithRepeatingConformers]:
        file_name = f"{mol_id}.pkl"
        files = os.listdir(conformers_dir)
        if file_name not in files:
            raise ValueError(f"Conformers for molecule {mol_id} not found")
        file_path = os.path.join(conformers_dir, file_name)
        with open(file_path, "rb") as f:
            mol = pickle.load(f)

        smiles = Chem.MolToSmiles(mol)

        return [MolWithRepeatingConformers(smiles, mol, mol_id, self.num_conformers)]

    def get(self, idx):
        df_ = pd.read_csv(self.data_file_path)
        df_idx = df_.iloc[idx]
        mol_id = df_idx["mol_id"]
        y_list = [df_idx[self.dataset_fields[1]]]
        smiles = df_idx["smiles"]
        assert os.path.exists(self.conformers_dir), "Conformers directory does not exist"
        mols = self._load_conformer(smiles, self.conformers_dir, mol_id)
        # logging.info('Featurizing conformer')
        data_list, conformers_list = self.featurizer.featurize(
            mols, y_list, self.num_conformers
        )  ## num_conformers is set in the config file
        # logging.info(f'Featurized {len(conformers)} conformer')

        # logging.info('Creating num_atoms_index')
        num_atoms_index = LargeConformerBasedDataset.create_num_atoms_index(conformers_list)
        # logging.info('Created num_atoms_index')
        return data_list[0], num_atoms_index[0]


class LargeConformerBasedDatasetNTrials(LargeConformerBasedDataset):
    def get(self, idx):
        sample = self._get_element(idx)
        mol_id = sample[:, 2].astype(str)[0].strip()
        y_list = [sample[:, 1].astype(float)]
        assert os.path.exists(self.conformers_dir), "Conformers directory does not exist"
        mols = load_conformer(self.conformers_dir, mol_id)

        # logging.info('Featurizing conformer')
        data_list, conformers_list = self.featurizer.featurize_n_times(
            mols, y_list, self.num_conformers, 10
        )
        # logging.info(f'Featurized {len(conformers)} conformer')

        # logging.info('Creating num_atoms_index')
        num_atoms_index = []
        for i in range(len(data_list[0])):
            num_atoms_index.append(
                LargeConformerBasedDataset.create_num_atoms_index([conformers_list[0][i]])
            )
        # logging.info('Created num_atoms_index')

        return data_list[0], num_atoms_index[0]


class GEOMDataset(Dataset):
    def __init__(self, mode: str, data_dir: str, config: Namespace, dataset_idx: int):
        super().__init__()

        self.target = config.target[dataset_idx]
        self.dataset_fields = ["smiles", self.target, "mol_id"]
        self.featurizer = MolGraphFeaturizer3D()
        # self.data_dir = f'{data_dir}/molecule_net'
        self.data_dir = data_dir
        self.dataset_name = config.dataset_name[dataset_idx]
        self.data_file_path = f"{self.data_dir}/{self.dataset_name}/{mode}.csv"
        self.num_conformers = config.num_conformers

        with open(os.path.join(self.data_dir, self.dataset_name, "summary.json"), "r") as f:
            molnet_summary = json.load(f)

        self.dataset_dict = molnet_summary

        self.n_samples = calculate_len(self.data_file_path)

    def len(self):
        return self.n_samples

    def get(self, idx):
        sample = self._get_element(idx)
        mol_id = sample["mol_id"][0].astype(str).strip()
        y_list = [sample[self.target][0].astype(float)]
        smiles = sample["smiles"][0].strip()
        mols = self._load_conformer(mol_id, smiles)

        # logging.info('Featurizing conformer')
        data_list, conformers_list = self.featurizer.featurize(mols, y_list, self.num_conformers)
        # logging.info(f'Featurized {len(conformers)} conformer')

        # logging.info('Creating num_atoms_index')
        num_atoms_index = LargeConformerBasedDataset.create_num_atoms_index(conformers_list)
        # logging.info('Created num_atoms_index')
        return data_list[0], num_atoms_index[0]

    def _get_element(self, idx):
        with open(self.data_file_path, "r") as file:
            header = file.readline()
            for line_idx, line in enumerate(file):
                if idx == line_idx:
                    csv_string = f"{header}{line.strip()}"
                    return pd.read_csv(StringIO(csv_string))[self.dataset_fields]

    def _load_conformer(self, mol_id, smiles) -> List[MolWithRepeatingConformers]:
        # logging.info(f'Loading conformer {mol_id} from {self.conformers_dir}')

        file_name = self.dataset_dict[smiles]["pickle_path"]
        pickle_path = os.path.join(self.data_dir, file_name)
        with open(pickle_path, "rb") as f:
            conf_dic = pickle.load(f)

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        conformers = [conf["rd_mol"].GetConformers()[0] for conf in conf_dic["conformers"]]
        for conformer in conformers:
            mol.AddConformer(conformer, assignId=True)
        return [MolWithRepeatingConformers(smiles, mol, mol_id, self.num_conformers)]
