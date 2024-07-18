import logging
import random
import re
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, Conformer
from tqdm import tqdm
import time
import numpy as np


class MolWithRepeatingConformers:
    def __init__(self, smiles, mol: Chem.Mol, idx: str, num_conformers: int):
        self.smiles = smiles
        self.mol = mol
        self.idx = idx
        self.num_conformers = num_conformers
        conformers = mol.GetConformers()
        conformer_indices = [i for i in range(len(conformers))]
        if len(conformers) == 0:
            raise ValueError(f"No conformers found for {smiles}")
        random.seed(1)
        if num_conformers > len(conformers):
            conformer_indices = random.choices(conformer_indices, k=num_conformers)
        elif num_conformers < len(conformers):
            conformer_indices = random.sample(conformer_indices, k=num_conformers)
        elif num_conformers < 1:
            raise ValueError(
                f"num_conformers must be greater or equal than 1, but got {num_conformers}"
            )
        self.conformer_indices = conformer_indices

    def get_conformers(self) -> List[Conformer]:
        return [self.mol.GetConformer(idx) for idx in self.conformer_indices]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["mol"] = self.mol.ToBinary()
        return state

    def __setstate__(self, state):
        self.mol = Chem.Mol(state["mol"])
        self.idx = state["idx"]
        self.smiles = state["smiles"]
        self.num_conformers = state["num_conformers"]
        self.conformer_indices = state["conformer_indices"]


class ConformersGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        smiles_list: List[str],
        mol_ids: List[str],
        num_conformers: int,
        prune_conformers: bool,
    ) -> List[Chem.Mol]:
        pass


class RDKitConformersGenerator(ConformersGenerator):
    def generate(
        self,
        smiles_list: List[str],
        mol_ids: List[str],
        num_conformers: int,
        prune_conformers: bool,
    ) -> List[Chem.Mol]:
        mols_with_conformers = [None for _ in range(len(smiles_list))]
        failed_mols = []
        with ProcessPoolExecutor() as executor:
            futures = RDKitConformersGenerator._submit_parallel_conformers_generation(
                executor, num_conformers, smiles_list, mol_ids, prune_conformers
            )
            logging.info("🥦 Starting generation...")
            logtime = []
            for future in tqdm(
                as_completed(futures), desc="Generating conformers", total=len(smiles_list)
            ):
                index = futures[future]
                try:
                    start_time = time.time()
                    mols_with_conformers[index] = future.result()
                    logtime.append(time.time() - start_time)
                except ValueError as exc:
                    # get message string from exception and parse it by whitespaces
                    failed_mol_id = str(exc).split()[-1]
                    failed_mols.append(failed_mol_id)
                    mols_with_conformers[index] = None
            print(f"Avg Runtime: {np.mean(logtime)}")
        print(f"failed_mols: {failed_mols}")

        return mols_with_conformers

    @staticmethod
    def _submit_parallel_conformers_generation(
        executor, num_conformers, smiles_list, mol_ids, prune_conformers
    ):
        futures = {}
        for i, smiles in tqdm(
            enumerate(smiles_list),
            desc="Submitting conformers generation tasks",
            total=len(smiles_list),
        ):
            futures[
                executor.submit(
                    RDKitConformersGenerator._conformers_by_mol,
                    smiles.strip(),
                    str(mol_ids[i]).strip(),
                    num_conformers,
                    prune_conformers,
                )
            ] = i
        return futures

    @staticmethod
    def _conformers_by_mol(
        smiles: str, idx: str, num_conformers: int, prune_conformers: bool
    ) -> MolWithRepeatingConformers:
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)
        if prune_conformers:
            AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers, pruneRmsThresh=0.5)
        else:
            AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers)
        idx = RDKitConformersGenerator.conformers_filename_prefix(idx)
        return MolWithRepeatingConformers(smiles, molecule, idx, num_conformers)

    @staticmethod
    def conformers_filename_prefix(idx: str) -> str:
        return re.sub("[!@#$%^&*(){};:,./<>?|`~=_+]", "_", idx)
