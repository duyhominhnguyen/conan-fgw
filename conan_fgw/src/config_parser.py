from jsonargparse import ArgumentParser
from typing import List
import yaml


def create_args_parser() -> ArgumentParser:
    """
    Creates and returns an ArgumentParser for parsing command-line arguments.

    This function sets up an ArgumentParser with the following arguments:
    - --num_epochs: Number of epochs for training (int)
    - --num_cuda_devices: Number of CUDA devices to use (int)
    - --use_wandb: Flag to use Weights and Biases for experiment tracking (bool)
    - --num_conformers: Number of conformers to generate (int)
    - --batch_size: Size of each training batch (int)
    - --experiment: Type of experiment to run (type)
    - --early_stopping.min_delta: Minimum change to qualify as an improvement for early stopping (float)
    - --early_stopping.patience: Number of epochs with no improvement after which training will be stopped (int)
    - --dataset_name: List of dataset names to use (list)
    - --target: List of target variables (list)
    - --dummy_size: Size of the dummy data (int)
    - --disable_distribution: Flag to disable distribution (bool)
    - --use_lr_finder: Flag to use learning rate finder (bool)
    - --learning_rate: Learning rate for the optimizer (float)
    - --prune_conformers: Flag to prune conformers (bool)
    - --max-iter: Maximum number of iterations for Bary Center (int, default=100)
    - --epsilon: Epsilon value for Bary Center (float, default=0.1)
    - --trade-off: Flag to enable trade-off for AUC-PRC (bool, default=False)

    Returns:
        ArgumentParser: Configured ArgumentParser instance.

    To view an example of config at "conan_fgw/config/schnet/property_regression/lipo/lipo_3_bc.yaml"
    """
    args_parser = ArgumentParser()
    args_parser.add_argument("--num_epochs", type=int)
    args_parser.add_argument("--num_cuda_devices", type=int)
    args_parser.add_argument("--use_wandb", type=bool)
    args_parser.add_argument("--num_conformers", type=int)
    args_parser.add_argument("--batch_size", type=int)
    args_parser.add_argument("--experiment", type=type)
    args_parser.add_argument("--early_stopping.min_delta", type=float)
    args_parser.add_argument("--early_stopping.patience", type=int)
    args_parser.add_argument("--dataset_name", type=list)
    args_parser.add_argument("--target", type=list)
    args_parser.add_argument("--dummy_size", type=int)
    args_parser.add_argument("--disable_distribution", type=bool)
    args_parser.add_argument("--use_lr_finder", type=bool)
    args_parser.add_argument("--learning_rate", type=float)
    args_parser.add_argument("--prune_conformers", type=bool)
    ## args Bary Center
    args_parser.add_argument("--max-iter", type=int, default=100)
    args_parser.add_argument("--epsilon", type=float, default=0.1)

    ## args trade off AUC-PRC (classification task only)
    args_parser.add_argument("--trade-off", type=bool, default=False)

    return args_parser
