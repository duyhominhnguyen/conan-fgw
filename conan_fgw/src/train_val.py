import logging
import os
import numpy as np
import torch
from pytorch_lightning.tuner.tuning import Tuner
import warnings
from sklearn.utils import class_weight
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from typing import Type, Union
from conan_fgw.src.config_parser import config_yaml_parser, cmd_args_parser
from conan_fgw.src.trainer import TrainerHolder
from conan_fgw.src.model.utils import load_dummy, seed_everything
from conan_fgw.src.utils import (
    build_logger,
    get_device,
    get_conan_fgw_pre_ckpt,
    AverageRuns,
    format_log_message_table,
)

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def create_model(
    dataset_class: Type,
    config: object,
    data_dir: str,
    dataset_idx: int,
    is_distributed: bool,
    model_name: str = "schnet",
    stage: str = "conan_fgw_pre",
) -> object:
    """
    Create and configure a machine learning model for classification or regression tasks.

    Args:
        dataset_class (Type): The class of the dataset being used.
        config (object): Configuration object containing experiment parameters.
        data_dir (str): Directory where the data is stored.
        dataset_idx (int): Index of the dataset.
        is_distributed (bool): Flag indicating whether the model training is distributed.
        model_name (str): Name of the model to be created. Available options are ["schnet", "visnet"].
        stage (str, optional): Stage of the experiment. Defaults to "conan_fgw_pre". Available options are ["conan_fgw_pre", "conan_fgw"]

    Returns:
        object: The configured model instance.
    """

    assert model_name in ["schnet", "visnet"]
    assert stage in ["conan_fgw_pre", "conan_fgw"]

    if "Classification" in config.experiment.model_class.__name__:
        train_ds = dataset_class("train", data_dir, config, dataset_idx)
        data_file = pd.read_csv(train_ds.data_file_path)
        target = train_ds.dataset_fields[1]
        gt = data_file[target].tolist()
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=[0, 1], y=gt)
        class_weights = torch.tensor([cw[1] / cw[0]])
        model = config.experiment.model_class(
            num_conformers=config.num_conformers,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            class_weights=class_weights,
            is_distributed=is_distributed,
            trade_off=config.trade_off,
            agg_weight=config.agg_weight,
        )
    else:
        ## Regression
        if stage == "conan_fgw_pre":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                model_name=model_name,
            )
        elif stage == "conan_fgw":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                model_name=model_name,
                is_distributed=is_distributed,
                max_iter=config.max_iter,
                epsilon=config.epsilon,
                agg_weight=config.agg_weight,
            )

    # Generating dummy data if the model is distributed
    if is_distributed:
        load_dummy(model, dataset=dataset_class("train", data_dir, config, dataset_idx))
    return model


def main():
    cmd_args = cmd_args_parser()
    cmd_args = cmd_args.parse_args()
    config_path = cmd_args.config_path
    data_dir = f"{cmd_args.data_root}/data"
    cuda_device = cmd_args.cuda_device
    number_of_runs = cmd_args.number_of_runs
    stage = cmd_args.stage
    run_name = cmd_args.run_name
    model_name = cmd_args.model_name
    run_id = cmd_args.run_id
    conan_fgw_pre_ckpt_dir = cmd_args.conan_fgw_pre_ckpt_dir
    verbose = cmd_args.verbose

    config_parser = config_yaml_parser()
    config = config_parser.instantiate_classes(config_parser.parse_path(f"{config_path}"))
    global logger
    logger_filename = os.path.join(
        cmd_args.logs_dir, "logs", run_name, run_id, f"run_{stage}", "log.txt"
    )

    logger = build_logger(
        logger_name="ConAN",
        logger_filename=logger_filename,
    )

    metric_logger = AverageRuns(config=config)

    # number_of_runs = number_of_runs if stage == "conan_fgw" else 1

    for run_idx in range(number_of_runs):
        run_idx += 1
        full_run_name = os.path.join(run_name, run_id, f"run_{stage}:" + str(run_idx))
        table = format_log_message_table(
            stage=f"{stage} @ Run {run_idx}/{number_of_runs}",
            checkpoint_path=os.path.join(cmd_args.checkpoints_dir, full_run_name),
            metric_path=os.path.join(cmd_args.logs_dir, "metrics", full_run_name),
            log_path=logger_filename,
        )

        dataset_class, datamodule_class = (
            config.experiment.dataset_class,
            config.experiment.datamodule_class,
        )

        dataset_idx = 0

        device, is_distributed = get_device(config, cuda_device)

        trainer_holder = TrainerHolder(
            config=config,  # Configuration object containing experiment parameters.
            is_distributed=is_distributed,  # Flag indicating if model training is distributed.
            device=device,  # Torch device on which the model will be trained.
            checkpoints_dir=cmd_args.checkpoints_dir,  # Directory where model checkpoints will be saved.
            logs_dir=cmd_args.logs_dir,  # Directory where logs will be saved.
            monitor_set="val",  # Subset used for monitoring during training.
        )

        data_module = datamodule_class(
            train_dataset=dataset_class("train", data_dir, config, dataset_idx),
            val_dataset=dataset_class("val", data_dir, config, dataset_idx),
            test_dataset=dataset_class("test", data_dir, config, dataset_idx),
            batch_size=config.batch_size,
            is_distributed=is_distributed,
        )

        model = create_model(
            dataset_class=dataset_class,  # Class representing the dataset.
            config=config,  # Configuration object containing experiment parameters.
            data_dir=data_dir,  # Directory path where the data is stored.
            dataset_idx=dataset_idx,  # Index of the dataset to be used.
            is_distributed=is_distributed,  # Flag indicating if model training is distributed.
            model_name=model_name,  # Name of the model to be created. Options: ["schnet", "visnet"]
            stage=stage,  # Stage of the experiment. Options: ["conan_fgw_pre", "conan_fgw"]
        )

        if stage == "conan_fgw":
            # Get the path to the conan_fgw_pre checkpoint for the fgw run
            conan_fgw_pre_ckpt_path = get_conan_fgw_pre_ckpt(
                conan_fgw_pre_ckpt_dir=conan_fgw_pre_ckpt_dir, run_idx=run_idx
            )
            # If an conan_fgw_pre checkpoint path is found, load the checkpoint
            if conan_fgw_pre_ckpt_path:
                checkpoint = torch.load(conan_fgw_pre_ckpt_path)
                model.load_state_dict(checkpoint["state_dict"])
                if verbose:
                    logger.info(
                        f"Load best model of conan_fgw_pre stage @: {conan_fgw_pre_ckpt_path}"
                    )
            else:
                if verbose:
                    logger.info(
                        f"Training ConAN-FGW from scratch without loading conan_fgw_pre checkpoint"
                    )

        trainer = trainer_holder.create_trainer(run_name=full_run_name, verbose=verbose)

        if config.use_lr_finder:
            tuner = Tuner(trainer)
            tuner.lr_find(model, datamodule=data_module)

        logger.info(table)
        trainer.fit(model=model, datamodule=data_module)
        metric_logger._register_metric(trainer, stage="train_val")
        trainer.test(model=model, datamodule=data_module, ckpt_path="best", verbose=verbose)
        metric_logger._register_metric(trainer, stage="test")

    runs_metrics = metric_logger.get_avg_metric()
    if is_distributed:
        if torch.distributed.get_rank() == 0:
            if stage == "conan_fgw":
                logger.info(runs_metrics)
            elif stage == "conan_fgw_pre":
                if verbose:
                    logger.info(runs_metrics)
    else:
        if stage == "conan_fgw":
            logger.info(runs_metrics)
        elif stage == "conan_fgw_pre":
            if verbose:
                logger.info(runs_metrics)


if __name__ == "__main__":
    seed_everything(5)
    main()
