import logging
import random
import string
from argparse import Namespace
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    Timer,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import SingleDeviceStrategy, DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.functional import mean_squared_error, auroc, mean_absolute_error
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import os

python_logger = logging.getLogger("trainer")


class TrainerHolder:
    def __init__(
        self,
        config: Namespace,
        is_distributed: bool,
        device: torch.device,
        checkpoints_dir: str,
        logs_dir: str,
        monitor_set: str = "val",
    ):
        self.config = config
        self.device = device
        self.logs_dir = os.path.join(logs_dir, "metrics")
        if "Classification" in config.experiment.model_class.__name__:
            if config.trade_off:
                self.metric_to_monitor = f"{monitor_set}_{TrainerHolder.classification_metric_name(config.trade_off)[-1]}"
            else:
                self.metric_to_monitor = f"{monitor_set}_{TrainerHolder.classification_metric_name(config.trade_off)[0]}"
        else:
            self.metric_to_monitor = f"val_{TrainerHolder.regression_metric_name()}"
        logging.info(f"🐸 Metric to monitor: {self.metric_to_monitor}")
        self.is_distributed = is_distributed
        self.checkpoints_dir = checkpoints_dir
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
        self.timer = Timer(verbose=True)
        self.monitor_set = monitor_set
        self.StochasticWeightAveraging = StochasticWeightAveraging(swa_lrs=1e-2)

    @staticmethod
    def regression_metric_name():
        # if "SOTABDE" in self.config.experiment.model_class.__name__:
        # return "mae"
        # else:
        return "rmse"

    @staticmethod
    def classification_metric_name(trade_off: bool = False):
        if trade_off:
            return ["auroc", "prc", "mean"]
        else:
            return ["auroc", "prc"]

    @staticmethod
    def regression_metric(predicted, expected):
        # if "SOTABDE" in self.config.experiment.model_class.__name__:
        # return mean_absolute_error(predicted, expected)
        # else:
        return mean_squared_error(predicted, expected, squared=True)

    @staticmethod
    def classification_metric(predicted, expected, trade_off: bool = False):
        expected = expected.long()
        auc_score = roc_auc_score(
            y_true=expected.cpu().detach().numpy(), y_score=predicted.cpu().detach().numpy()
        )
        precision, recall, thresholds = precision_recall_curve(
            y_true=expected.cpu().detach().numpy(), probas_pred=predicted.cpu().detach().numpy()
        )
        prc_score = auc(recall, precision)
        if trade_off:
            metric = [auc_score, prc_score, (auc_score + prc_score) / 2.0]
        else:
            metric = [auc_score, prc_score]
        return metric

    def create_trainer(self, run_name: str, use_distributed_sampler: bool = True) -> pl.Trainer:
        if "Classification" in self.config.experiment.model_class.__name__:
            task = "classification"
        else:
            task = "regression"
        logging.info(f"🔑 Callback task: {task}")
        callbacks = [
            self.early_stopping_callback(use_loss=True),
            self.checkpoint_callback(run_name, task=task),
            self.lr_monitor,
            self.timer,
            # self.StochasticWeightAveraging
            # self.gradient_accumulator_callback()]
        ]

        return pl.Trainer(
            max_epochs=self.config.num_epochs,
            logger=self.create_logger(run_name),
            log_every_n_steps=self.config.batch_size,
            callbacks=callbacks,
            strategy=self.training_strategy(),
            gradient_clip_val=1.0,
            # gradient_clip_algorithm="value",
            default_root_dir=self.checkpoints_dir,
            num_sanity_val_steps=-1,
            # use_distributed_sampler=use_distributed_sampler
        )

    def create_logger(self, experiment_name=None) -> [pl.loggers.Logger]:
        if not experiment_name:
            experiment_name = "".join(random.choice(string.ascii_lowercase) for _ in range(15))
        loggers = [CSVLogger(self.logs_dir, experiment_name)]
        return loggers

    def early_stopping_callback(self, use_loss: bool = False) -> EarlyStopping:
        if use_loss:
            obj_monitor = "val_loss"
            mode = "min"
        else:
            obj_monitor = self.metric_to_monitor
            mode = "max"

        return EarlyStopping(
            monitor=obj_monitor,
            min_delta=self.config.early_stopping.min_delta,
            patience=self.config.early_stopping.patience,
            mode=mode,
            verbose=True,
            check_finite=True,
        )

    def checkpoint_callback(self, run_name: str, task: str = "regression") -> ModelCheckpoint:
        if task == "regression":
            return ModelCheckpoint(
                dirpath=f"{self.checkpoints_dir}/{run_name}",  # Directory where the checkpoints will be saved
                filename="{epoch}-{step}-{val_rmse:.2f}",  # Checkpoint file name format
                verbose=True,  # Print a message when a new best checkpoint is saved
                monitor=self.metric_to_monitor,  # Metric to monitor for saving best checkpoints
                mode="min",  # Minimize the monitored metric (use "max" for metrics like accuracy)
                save_last=True,  # Save a checkpoint for the last epoch
            )
        elif task == "classification":
            if "mean" in self.metric_to_monitor:
                if self.monitor_set == "train":
                    return ModelCheckpoint(
                        dirpath=f"{self.checkpoints_dir}/{run_name}",  # Directory where the checkpoints will be saved
                        filename="{epoch}-{step}-{train_mean:.2f}",  # Checkpoint file name format
                        verbose=True,  # Print a message when a new best checkpoint is saved
                        monitor=self.metric_to_monitor,  # Metric to monitor for saving best checkpoints
                        mode="max",  # Minimize the monitored metric (use "max" for metrics like accuracy)
                        save_last=True,  # Save a checkpoint for the last epoch
                    )
                else:
                    return ModelCheckpoint(
                        dirpath=f"{self.checkpoints_dir}/{run_name}",  # Directory where the checkpoints will be saved
                        filename="{epoch}-{step}-{val_mean:.2f}",  # Checkpoint file name format
                        verbose=True,  # Print a message when a new best checkpoint is saved
                        monitor=self.metric_to_monitor,  # Metric to monitor for saving best checkpoints
                        mode="max",  # Minimize the monitored metric (use "max" for metrics like accuracy)
                        save_last=True,  # Save a checkpoint for the last epoch
                    )
            else:
                if self.monitor_set == "train":
                    return ModelCheckpoint(
                        dirpath=f"{self.checkpoints_dir}/{run_name}",  # Directory where the checkpoints will be saved
                        filename="{epoch}-{step}-{train_auroc:.2f}",  # Checkpoint file name format
                        verbose=True,  # Print a message when a new best checkpoint is saved
                        monitor=self.metric_to_monitor,  # Metric to monitor for saving best checkpoints
                        mode="max",  # Minimize the monitored metric (use "max" for metrics like accuracy)
                        save_last=True,  # Save a checkpoint for the last epoch
                    )
                else:
                    return ModelCheckpoint(
                        dirpath=f"{self.checkpoints_dir}/{run_name}",  # Directory where the checkpoints will be saved
                        filename="{epoch}-{step}-{val_auroc:.2f}",  # Checkpoint file name format
                        verbose=True,  # Print a message when a new best checkpoint is saved
                        monitor=self.metric_to_monitor,  # Metric to monitor for saving best checkpoints
                        mode="max",  # Minimize the monitored metric (use "max" for metrics like accuracy)
                        save_last=True,  # Save a checkpoint for the last epoch
                    )

    @staticmethod
    def gradient_accumulator_callback() -> GradientAccumulationScheduler:
        return GradientAccumulationScheduler(scheduling={0: 10, 4: 5, 10: 1})

    def training_strategy(self, is_fgw: bool = False):
        if self.device.type == "cuda":
            if self.is_distributed:
                return "ddp_find_unused_parameters_true"
            else:
                return SingleDeviceStrategy(self.device, accelerator="cuda")
        elif self.device.type == "cpu":
            return SingleDeviceStrategy(self.device, accelerator="cpu")
        elif self.device.type == "mps":
            # return SingleDeviceStrategy(self.device, accelerator='mps')
            # A lot of operations are not supported on M1 yet
            return SingleDeviceStrategy(torch.device("cpu"), accelerator="cpu")
