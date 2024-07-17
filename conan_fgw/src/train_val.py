import logging
import os
import uuid
import numpy as np
import torch
import wandb
from jsonargparse import ArgumentParser
from prettytable import PrettyTable
from pytorch_lightning.tuner.tuning import Tuner
import warnings
import random
from sklearn.utils import class_weight
import pandas as pd
from tqdm import tqdm
import datetime
import torch.distributed as dist
import time

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from conan_fgw.src.config_parser import create_args_parser
from conan_fgw.src.trainer import TrainerHolder
from conan_fgw.src.model.utils import load_dummy, seed_everything
from conan_fgw.src.utils import build_logger

torch.set_float32_matmul_precision("medium")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
# python_logger = logging.getLogger("main")


def create_model(
    dataset_class,
    config,
    data_dir,
    dataset_idx,
    is_distributed,
    run_name,
    model_name,
    stage: str = "initial",
):

    if "Classification" in config.experiment.model_class.__name__:
        train_ds = dataset_class("train", data_dir, config, dataset_idx)
        data_file = pd.read_csv(train_ds.data_file_path)
        target = train_ds.dataset_fields[1]
        gt = data_file[target].tolist()
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=[0, 1], y=gt)
        class_weights = torch.tensor([cw[1] / cw[0]])
        if stage == "initial":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                class_weights=class_weights,
                is_distributed=is_distributed,
                trade_off=config.trade_off,
                run_name=run_name,
            )
        elif stage == "fgw":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                class_weights=class_weights,
                is_distributed=is_distributed,
                trade_off=config.trade_off,
                run_name=run_name,
            )
    else:
        ## Regression
        if stage == "initial":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                model_name=model_name,
            )
        elif stage == "fgw":
            model = config.experiment.model_class(
                num_conformers=config.num_conformers,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                is_distributed=is_distributed,
                max_iter=config.max_iter,
                epsilon=config.epsilon,
                run_name=run_name,
                model_name=model_name,
            )

    ## generating dummy
    if is_distributed:
        load_dummy(model, dataset=dataset_class("train", data_dir, config, dataset_idx))
    logging.info(f"Model is created with {stage} stage")
    return model


def get_initial_ckpt(initial_ckpt_dir: str, run_idx: str):
    initial_ckpt_dir = os.path.join(initial_ckpt_dir, f"run_initial:{run_idx}")
    ckpts = os.listdir(initial_ckpt_dir)
    initial_ckpt_path = None
    for ckpt in ckpts:
        if ckpt.startswith("epoch"):
            initial_ckpt_path = os.path.join(initial_ckpt_dir, ckpt)
            return initial_ckpt_path
    return initial_ckpt_path


def main():
    cmd_args_parser = ArgumentParser()
    cmd_args_parser.add_argument("--config_path", type=str)
    cmd_args_parser.add_argument("--cuda_device", type=int)
    cmd_args_parser.add_argument("--data_root", type=str)
    cmd_args_parser.add_argument("--number_of_runs", type=int, default=3)
    cmd_args_parser.add_argument("--checkpoints_dir", type=str)
    cmd_args_parser.add_argument("--logs_dir", type=str)
    cmd_args_parser.add_argument("--run_name", type=str)
    cmd_args_parser.add_argument("--stage", type=str, default="initial")
    cmd_args_parser.add_argument("--run_id", type=str)
    cmd_args_parser.add_argument("--model_name", type=str, default="schnet")
    cmd_args_parser.add_argument("--initial_ckpt_dir", default=None)

    cmd_args = cmd_args_parser.parse_args()

    config_path = cmd_args.config_path

    data_root = cmd_args.data_root
    data_dir = f"{data_root}/data"
    cuda_device = cmd_args.cuda_device
    number_of_runs = cmd_args.number_of_runs
    stage = cmd_args.stage
    run_name = cmd_args.run_name
    model_name = cmd_args.model_name
    run_id = cmd_args.run_id
    initial_ckpt_dir = cmd_args.initial_ckpt_dir

    config_parser: ArgumentParser = create_args_parser()

    config = config_parser.instantiate_classes(config_parser.parse_path(f"{config_path}"))
    for run_idx in range(number_of_runs):

        logging.info(f"Run: {run_idx+1}/{number_of_runs}")

        dataset_class, datamodule_class = (
            config.experiment.dataset_class,
            config.experiment.datamodule_class,
        )

        dataset_idx = 0

        device, is_distributed = get_device(config, cuda_device)
        logging.info(f"device: {device} | is_distributed: {is_distributed}")

        trainer_holder = TrainerHolder(
            config=config,
            is_distributed=is_distributed,
            device=device,
            checkpoints_dir=cmd_args.checkpoints_dir,
            logs_dir=cmd_args.logs_dir,
            monitor_set="val",
        )

        data_module = datamodule_class(
            dataset_class("train", data_dir, config, dataset_idx),
            dataset_class("val", data_dir, config, dataset_idx),
            dataset_class("test", data_dir, config, dataset_idx),
            config.batch_size,
            is_distributed,
        )

        run_val_mae, run_test_rmse = [], []

        run_name = run_name + "/" + run_id

        model = create_model(
            dataset_class=dataset_class,
            config=config,
            data_dir=data_dir,
            dataset_idx=dataset_idx,
            is_distributed=is_distributed,
            run_name=run_name,
            model_name=model_name,
            stage=stage,
        )

        if stage == "fgw":
            initial_ckpt_path = get_initial_ckpt(
                initial_ckpt_dir=initial_ckpt_dir, run_idx=run_idx
            )
            if initial_ckpt_path:
                checkpoint = torch.load(initial_ckpt_path)
                logging.info(f"Load best model of initial stage @: {initial_ckpt_path}")
                model.load_state_dict(checkpoint["state_dict"])

        trainer = trainer_holder.create_trainer(
            run_name=run_name + f"/run_{stage}:" + str(run_idx)
        )

        if config.use_lr_finder:
            logging.info("Starting learning rate finder")
            tuner = Tuner(trainer)
            tuner.lr_find(model, datamodule=data_module)
            # if stage == "train":
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

        if stage == "initial":
            initial_ckpt_path = trainer.checkpoint_callback.best_model_path
            logging.info(f"Found best model of initial stage @: {initial_ckpt_path}")


def get_device(config, cuda_device):
    if torch.cuda.is_available():
        is_distributed = torch.cuda.device_count() > 1 and not config.disable_distribution
        if is_distributed:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{cuda_device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        is_distributed = False
    else:
        device = torch.device("cpu")
        is_distributed = False
    return device, is_distributed


if __name__ == "__main__":
    seed_everything(5)
    main()
