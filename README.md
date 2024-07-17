# Structure-Aware E(3)-Invariant Molecular Conformer Aggregation Networks
:fire: :fire: This repository contains PyTorch implementation for ICML 2024 paper: **Structure-Aware E(3)-Invariant Molecular Conformer Aggregation Networks [[arXiv]](https://arxiv.org/abs/2402.01975)**.

A molecule's 2D representation consists of its atoms, their attributes, and the molecule's covalent bonds. A 3D (geometric) representation of a molecule is called a conformer and consists of its atom types and Cartesian coordinates. Every conformer has a potential energy, and the lower this energy, the more likely it occurs in nature. Most existing machine learning methods for molecular property prediction consider either 2D molecular graphs or 3D conformer structure representations in isolation. Inspired by recent work on using ensembles of conformers in conjunction with 2D graph representations, we propose E(3)-invariant molecular conformer aggregation networks. The method integrates a molecule's 2D representation with that of multiple of its conformers. Contrary to prior work, we propose a novel 2D--3D aggregation mechanism based on a differentiable solver for the Fused Gromov-Wasserstein Barycenter problem and the use of an efficient online conformer generation method based on distance geometry. We show that the proposed aggregation mechanism is E(3) invariant and provides an efficient GPU implementation. Moreover, we demonstrate that the aggregation mechanism helps to significantly outperform state-of-the-art property prediction methods on established datasets.
![Overview figure](figs/ala.png)

## Table of Contents

- [Update](#update)
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)

## Update
- **May 2024**: We release 1st version of codebase.

## Introduction
This repository provides implementations for E(3)-invariant molecular conformer aggregation networks on a collection of 6 benchmark datasets designed to address a wide range of graph-related tasks such as classification and regression. Our implementations are built on state-of-the-art deep learning frameworks and are designed to be easily extensible and customizable.

The repository is structured as follows:

- **data/**: This directory contains scripts and utilities for downloading and preprocessing benchmark datasets.
- **outputs/**: This directory contains processes' outcome including logs.
- **models/**: This directory contains processes' outcome including checkpoints.
- **conan_fgw/script**: This directory is intended to store experimental scripts.
- **conan_fgw/src**: This directory contains the source code for training, evaluating, and visualizing models.
- **conan_fgw/config**: This directory is intended to store experimental configurations.
- **README.md**: This file contains information about the project, including installation instructions, usage examples, and a description of the repository structure.
- **environment.yml**: This file lists all Python dependencies required to run the project.
- **.gitignore**: This file specifies which files and directories should be ignored by Git version control.

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
After finishing the download process, please put them into the directory **/data**.

## Usage
The project focuses on leveraging four MoleculeNet datasets: Lipo, ESOL, FreeSolv, BACE, and two CoV-2 datasets. All relevant data is stored within the **/data** directory. To configure the settings for each dataset, corresponding configuration files are provided in the **conan_fgw/config/** folder.

To reproduce experiments, please refer:
```bash
bash conan_fgw/script/run.sh
```

For example, to experiment with **ConAN** using the **SchNet** network as the backbone on the **Lipo** dataset, the script [run.sh](./conan_fgw/script/run.sh) should be as follows:

```bash
## conan_fgw/script/run.sh
# Check if the .env file exists
if [ ! -f .env ]
then
  # If the .env file exists, export the environment variables defined in it
  export $(cat .env | xargs)
fi

# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
# Get the current date and time in the format YYYY-MM-DD-T
DATE=$(date +"%Y-%m-%d-%T")

# Define variables for the model, task, dataset, number of conformers, and number of runs
model=schnet                      
task=property_regression
ds=lipo
n_cfm=3
runs=3

# Set the visible CUDA devices to the first GPU for initial training stage
export CUDA_VISIBLE_DEVICES=0
# Run the initial training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=initial \
        --model_name=${model} \
        --run_id=$DATE

# Set the visible CUDA devices to GPUs 0, 1, 2, and 3 for using Distributed Data Parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the FGW (Fused Gromov-Wasserstein) training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm\_bc.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=fgw \
        --model_name=${model} \
        --run_id=$DATE \
        --initial_ckpt_dir=${WORKDIR}/models/$model\_$ds\_$n_cfm/$DATE
```

For other experiments, we need to change the following group of arguments:

```bash
model=<backbone_gnn_model>    ## Two backbone GNNs: schnet and visnet                      
task=<molecular_task>         ## Two main molecular tasks: property_regression and classification
ds=<dataset_name>             ## Four datasets for property_regression and two datasets for classification tasks
n_cfm=3                       ## Number of conformers used for extraction by 3D message passing networks
runs=3                        ## Number of runs for general evaluation
```

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
