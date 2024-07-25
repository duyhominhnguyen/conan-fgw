<h4 align="center">
    <img alt="ConAN logo" src="docs/images/conan.png" style="width: 100%;">
</h4>

<p align="center">
    <a href="https://lightning.ai/docs/pytorch/stable">
        <img alt="Pytorch Lightning" src="https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning">
    </a>
    <a href="https://star-history.com/#duyhominhnguyen/conan-fgw">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/duyhominhnguyen/conan-fgw?style=flat-square">
    </a>
    <a href="https://github.com/duyhominhnguyen/conan-fgw/issues">
        <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/duyhominhnguyen/conan-fgw?style=flat-square">
    </a>
    <a href="https://opensource.org/license/MIT">
        <img alt="License" src="https://img.shields.io/github/license/duyhominhnguyen/conan-fgw">
    </a>
</p>

<h1>
    <p align="center">
        Structure-Aware E(3)-Invariant Molecular Conformer Aggregation Networks
    </p>
</h1>

:fire: :fire: This repository contains PyTorch implementation for our paper: **Structure-Aware E(3)-Invariant Molecular Conformer Aggregation Networks (ICML 2024) [[arXiv]](https://arxiv.org/abs/2402.01975) [[Poster](https://github.com/duyhominhnguyen/conan-fgw/blob/main/docs/ICML-2024-ConAN_final.pdf)]**.

![Overview figure](figs/ala.png)

## Table of Contents

- [Update](#update)
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)

## Update
:mega: **17 July 2024**: We release 1st version of codebase.

## Introduction
We provide implementations for E(3)-invariant molecular conformer aggregation networks (**ConAN**) on a collection of six benchmark datasets related to **molecular property prediction** and **molecular classification**. Our model builds on state-of-the-art deep learning frameworks and is designed to be easily extensible and customizable.

The repository is structured as follows:

- ```data/```: This directory contains scripts and utilities for downloading and preprocessing benchmark datasets.
- ```outputs/```: This directory contains processes' outcome including logs.
- ```models/```: This directory contains processes' outcome including checkpoints.
- ```conan_fgw/script```: This directory is intended to store experimental scripts.
- ```conan_fgw/src```: This directory contains the source code for training, evaluating, and visualizing models.
- ```conan_fgw/config```: This directory is intended to store experimental configurations.
- ```README.md```: This file contains information about the project, including installation instructions, usage examples, and a description of the repository structure.
- ```environment.yml```: This file lists all Python dependencies required to run the project.
- ```.gitignore```: This file specifies which files and directories should be ignored by Git version control.

## Installation

To re-produce this project, you will need to have the following dependencies installed:
- Ubuntu 18.04.6 LTS
- CUDA Version: 11.7
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3
- [PyTorch](https://pytorch.org/) (version 2.0 or later)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)

After installing Miniconda, you can create a new environment and install the required packages using the following commands:

```bash
conda create -n conan python=3.9
conda activate conan
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
conda env update -n conan --file environment.yaml
```

## Data
To refer benchmark datasets, please get access this link and download [here](https://mega.nz/folder/X9VEXb7D#xv6fXIon_00tgevNMZn73A).
After finishing the download process, please put them into the directory ```/data```.

## Quickstart
The project focuses on leveraging four MoleculeNet datasets: Esol, Lipo, FreeSolv, BACE, and two CoV-2 datasets. All relevant data is stored within the ```/data``` directory. To configure the settings for each dataset, corresponding configuration files are provided in the ```conan_fgw/config/``` folder.

To reproduce experiments, please refer:
```bash
bash conan_fgw/script/run.sh
```

For example, to experiment with **ConAN** using the **SchNet** network as the backbone on the **Esol** dataset, the script [run.sh](./conan_fgw/script/run.sh) should be as follows:

First, define variables for the model, task, dataset, number of conformers, and number of runs:

```bash
## conan_fgw/script/run.sh
model=schnet                    ## Message Passing Backbone: schnet OR visnet
task=property_regression        ## Molecular tasks: property_regression OR classification
ds=esol                         ## Dataset name
n_cfm_conan_fgw_pre=5           ## Number of conformers used in conan-fgw pretraining stage
n_cfm_conan_fgw=5               ## Number of conformers used in conan-fgw training stage
runs=5                          ## Number of runs for general evaluation
```

**Note**: Please refer to the configurations for a certain experiment. They should be available at `conan_fgw/config/<selected_model>/<molecular_task>/<dataset_name>`. In this case, there are two configuration YAML files named `esol_5.yaml` and `esol_5_bc.yaml` in the directory `conan_fgw/config/schnet/property_regression/esol/`:

<details>
  <summary><i>esol_5.yaml</i></summary>

```yml
## esol_5.yaml
disable_distribution: true  # Disable distribution of the data across multiple devices or nodes.
dataset_name: ['esol']  # List of dataset names to be used. Here, it's the ESOL dataset.
dummy_size: -1  # Size of a dummy dataset for testing. -1 indicates not using a dummy dataset.
target: ['measured_log_sol']  # Target property to predict, here it's the measured log solubility.
num_conformers: 5  # Number of conformers to generate per molecule.
prune_conformers: false  # Whether to prune conformers to a smaller set.
batch_size: 96  # Number of samples per batch during training.
experiment: conan_fgw.src.experiments.SOTAExperiment  # Path to the experiment class used for training.
num_epochs: 150  # Total number of training epochs.
early_stopping:  # Early stopping configuration to prevent overfitting.
  min_delta: 0.0001  # Minimum change in the monitored metric to qualify as an improvement.
  patience: 50  # Number of epochs with no improvement after which training will be stopped.
learning_rate: 0.001  # Initial learning rate for training.
use_lr_finder: false  # Whether to use a learning rate finder to automatically adjust the learning rate.
use_wandb: false  # Whether to use Weights & Biases for experiment tracking.
```
</details>

<details>
  <summary><i>esol_5_bc.yaml</i></summary>

```yml
## esol_5_bc.yaml
disable_distribution: false  # Whether to disable distribution of the data across multiple devices or nodes.
dataset_name: ['esol']  # List of dataset names to be used. Here, it's the ESOL dataset.
dummy_size: -1  # Size of a dummy dataset for testing. -1 indicates not using a dummy dataset.
target: ['measured_log_sol']  # Target property to predict, here it's the measured log solubility.
num_conformers: 5  # Number of conformers to generate per molecule.
prune_conformers: false  # Whether to prune conformers to a smaller set.
batch_size: 24  # Number of samples per batch during training.
experiment: conan_fgw.src.experiments.SOTAExperimentBaryCenter  # Path to the experiment class used for training.
num_epochs: 80  # Total number of training epochs.
early_stopping:  # Early stopping configuration to prevent overfitting.
  min_delta: 0.0001  # Minimum change in the monitored metric to qualify as an improvement.
  patience: 50  # Number of epochs with no improvement after which training will be stopped.
learning_rate: 0.0005  # Initial learning rate for training.
use_lr_finder: false  # Whether to use a learning rate finder to automatically adjust the learning rate.
use_wandb: false  # Whether to use Weights & Biases for experiment tracking.
agg_weight: 0.2  # Aggregation weight for combining different terms or losses.
```
</details>

then, the rest of the bash script follows:

1. Run the ConAN-FGW pretraining stage
```bash
export CUDA_VISIBLE_DEVICES=0
python conan_fgw/src/train_val.py \
    --config_path=${WORKDIR}/conan_fgw/config/${model}/${task}/${ds}/${ds}_${n_cfm}_conan_fgw_pre.yaml \
    --cuda_device=0 \
    --data_root=${WORKDIR} \
    --number_of_runs=${runs} \
    --checkpoints_dir=${WORKDIR}/models \
    --logs_dir=${WORKDIR}/outputs \
    --run_name=${model}_${ds}_${n_cfm}_conan_fgw_pre \
    --stage=conan_fgw_pre \
    --model_name=${model} \
    --run_id=${DATE}
```
2. Run the ConAN-FGW training stage

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python conan_fgw/src/train_val.py \
    --config_path=${WORKDIR}/conan_fgw/config/${model}/${task}/${ds}/${ds}_${n_cfm}_conan_fgw_bc.yaml \
    --cuda_device=0 \
    --data_root=${WORKDIR} \
    --number_of_runs=${runs} \
    --checkpoints_dir=${WORKDIR}/models \
    --logs_dir=${WORKDIR}/outputs \
    --run_name=${model}_${ds}_${n_cfm}_conan_fgw \
    --stage=conan_fgw \
    --model_name=${model} \
    --run_id=${DATE} \
    --conan_fgw_pre_ckpt_dir=${WORKDIR}/models/${model}_${ds}_${n_cfm}_conan_fgw_pre/${DATE}
```

<details>
  <summary><b>Full Script</b></summary>

```bash
## conan_fgw/script/run.sh
## Set the working directory to the current directory
export WORKDIR=$(pwd)
## Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
## Get the current date and time in the format YYYY-MM-DD-HH-MM-SS
DATE=$(date +"%Y-%m-%d-%H-%M-%S")
## Set the visible CUDA devices to the first GPU for conan_fgw_pre training stage
export CUDA_VISIBLE_DEVICES=0
## Run the conan_fgw_pre training stage
python conan_fgw/src/train_val.py \
    --config_path=${WORKDIR}/conan_fgw/config/${model}/${task}/${ds}/${ds}_${n_cfm}_conan_fgw_pre.yaml \
    --cuda_device=0 \
    --data_root=${WORKDIR} \
    --number_of_runs=${runs} \
    --checkpoints_dir=${WORKDIR}/models \
    --logs_dir=${WORKDIR}/outputs \
    --run_name=${model}_${ds}_${n_cfm}_conan_fgw_pre \
    --stage=conan_fgw_pre \
    --model_name=${model} \
    --run_id=${DATE}
## Set the visible CUDA devices to GPUs 0, 1, 2, and 3 for using Distributed Data Parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
## Run the FGW (Fused Gromov-Wasserstein) training stage
python conan_fgw/src/train_val.py \
    --config_path=${WORKDIR}/conan_fgw/config/${model}/${task}/${ds}/${ds}_${n_cfm}_conan_fgw_bc.yaml \
    --cuda_device=0 \
    --data_root=${WORKDIR} \
    --number_of_runs=${runs} \
    --checkpoints_dir=${WORKDIR}/models \
    --logs_dir=${WORKDIR}/outputs \
    --run_name=${model}_${ds}_${n_cfm}_conan_fgw \
    --stage=conan_fgw \
    --model_name=${model} \
    --run_id=${DATE} \
    --conan_fgw_pre_ckpt_dir=${WORKDIR}/models/${model}_${ds}_${n_cfm}_conan_fgw_pre/${DATE}
```

</details>

For your reference, we provide an abstract of two model classes **SchNet** and **ViSNet** related to the ConAN-FGW model initialization and calculation for both ConAN-FGW ```pretraining``` and ```training``` stages:

<details>
  <summary><b>SchNet</b></summary>

```python
## conan_fgw/src/model/graph_embeddings/schnet_no_sum.py
from torch_geometric.nn import SchNet # The SchNet class used in ConAN is an extension of the SchNet class of torch_geometric
class SchNetNoSum(SchNet):
  def __init__(
      self,
      device, # The device on which the model will run (e.g., CPU or GPU).
      hidden_channels: int = 128, # Number of hidden channels (default: 128).
      num_filters: int = 128, # Number of filters (default: 128).
      num_interactions: int = 6, # Number of interaction blocks (default: 6).
      num_gaussians: int = 50,  # Number of Gaussians for distance expansion (default: 50).
      cutoff: float = 10.0, # Cutoff distance for interactions (default: 10.0).
      interaction_graph: Optional[Callable] = None, # Optional callable for defining the interaction graph.
      max_num_neighbors: int = 32,  # Maximum number of neighbors for each atom (default: 32).
      readout: str = "add", # Readout function, default is "add".
      dipole: bool = False, # Whether to include dipole moment prediction (default: False).
      mean: Optional[float] = None, # Mean and standard deviation for normalization.
      std: Optional[float] = None,  
      atomref: OptTensor = None,  # Atomic reference values for target properties.
      use_covalent: bool = False, # Whether to use covalent bond information (default: False).
      use_readout: bool = True, # Whether to use the readout layer (default: True).
  ):
    ## Initialization
  def forward(
    self,
    z: Tensor,  # Atomic numbers of the atoms.
    pos: Tensor,  # Coordinates of the atoms.
    batch: OptTensor = None,  # Batch indices for separating molecules.
    data_batch=None # Additional data, such as covalent bonds attributes.
  ) -> Tensor:
    ## Forward Pass without Barycenter Calculation
    ## Returns: Tensor containing the computed features for each molecule.
  def forward_3d_bary(
    self,
    z: Tensor,  # Atomic numbers of the atoms.
    pos: Tensor,  # Coordinates of the atoms.
    batch: OptTensor = None,  # Batch indices for separating molecules.
    data_batch=None # Additional data, such as covalent bonds attributes.
  ) -> Tensor
    ## Forward Pass with Bary Center Calculation
    ## Returns: Two tensors, one for standard 3D aggregation and one for barycenter aggregation.
```
</details>


<details>
  <summary><b>ViSNet</b></summary>

```python
## conan_fgw/src/model/graph_embeddings/visnet.py
from torch_geometric.nn.models.visnet import ViSNet as NaiveViSNet # The ViSNet class used in ConAN is an extension of the ViSNet class of torch_geometric
class ViSNet(NaiveViSNet):
  def __init__(
    self,
    device, # The device on which the model will run (e.g., CPU or GPU).
    hidden_channels: int, # Number of hidden channels in the model.
    cutoff: float = 5.0 # Distance cutoff for interaction graph construction.
  ):
    ## Initialization
  def forward(
    self,
    z: Tensor,  # Atomic numbers of the atoms.
    pos: Tensor,  # Coordinates of the atoms.
    batch: OptTensor = None,  # Batch indices for separating molecules.
    data_batch=None # Additional data, such as covalent bonds attributes.
  ) -> Tensor:
    ## Forward Pass without Barycenter Calculation
    ## Returns: Tensor containing the computed features for each molecule.
  def forward_3d_bary(
    self,
    z: Tensor,  # Atomic numbers of the atoms.
    pos: Tensor,  # Coordinates of the atoms.
    batch: OptTensor = None,  # Batch indices for separating molecules.
    data_batch=None # Additional data, such as covalent bonds attributes.
  ) -> Tensor
    ## Forward Pass with Bary Center Calculation
    ## Returns: Two tensors, one for standard 3D aggregation and one for barycenter aggregation.
```
</details>

## Citation
Please cite this paper if it helps your research:
```bibtex
@article{nguyen2024structure,
  title={Structure-Aware E (3)-Invariant Molecular Conformer Aggregation Networks},
  author={Nguyen, Duy MH and Lukashina, Nina and Nguyen, Tai and Le, An T and Nguyen, TrungTin and Ho, Nhat and Peters, Jan and Sonntag, Daniel and Zaverkin, Viktor and Niepert, Mathias},
  journal={International Conference on Machine Learning},
  year={2024}
}
```
