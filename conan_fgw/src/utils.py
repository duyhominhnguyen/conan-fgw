import logging
import logging.handlers
import os
import torch
from prettytable import PrettyTable
from conan_fgw.src.trainer import TrainerHolder
import numpy as np


def build_logger(logger_name: str, logger_filename: str):
    global handler
    parent_dir = os.path.dirname(logger_filename)
    os.makedirs(parent_dir, exist_ok=True)
    formatter = logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.handlers.TimedRotatingFileHandler(logger_filename, when="D", utc=True)
    handler.setFormatter(formatter)

    for name, item in logging.root.manager.loggerDict.items():
        if isinstance(item, logging.Logger):
            item.addHandler(handler)
    return logger


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


def get_initial_ckpt(initial_ckpt_dir: str, run_idx: str):
    initial_ckpt_dir = os.path.join(initial_ckpt_dir, f"run_initial:{run_idx}")
    ckpts = os.listdir(initial_ckpt_dir)
    initial_ckpt_path = None
    for ckpt in ckpts:
        if ckpt.startswith("epoch"):
            initial_ckpt_path = os.path.join(initial_ckpt_dir, ckpt)
            return initial_ckpt_path
    return initial_ckpt_path


class AverageRuns:
    def __init__(self, config):
        self.config = config
        self.stats_table = PrettyTable()
        task = (
            "classfication"
            if "Classification" in config.experiment.model_class.__name__
            else "regression"
        )
        if task == "regression":
            metric_name = TrainerHolder.regression_metric_name()
            self.stats_table.field_names = ["dataset", f"val_{metric_name}", f"test_{metric_name}"]
        elif task == "classification":
            metric_name = TrainerHolder.classification_metric_name(config.trade_off)
            self.stats_table.field_names = ["dataset"]
            set_metrics = []
            for s in ["val", "test"]:
                for i in range(len(metric_name)):
                    set_metrics.append(f"{s}_{metric_name[i]}")
            self.stats_table.field_names += set_metrics
        else:
            ValueError(f"No implementation of task {task}")

        ## monitor metric follows checkpoint_callback
        self.monitor_metrics = {}
        for i in range(1, len(self.stats_table.field_names)):
            m_name = self.stats_table.field_names[i]
            self.monitor_metrics[m_name] = []

    def _register_metric(self, trainer, stage: str = "train_val"):
        for i in range(1, len(self.stats_table.field_names)):
            m_name = self.stats_table.field_names[i]
            if stage == "train_val":
                if m_name.startswith("val"):
                    metric = trainer.callback_metrics[m_name]
                    if torch.is_tensor(metric):
                        metric = metric.item()
                    metric = round(metric, 3)
                    self.monitor_metrics[m_name].append(metric)
                else:
                    continue
            elif stage == "test":
                if m_name.startswith("test"):
                    metric = trainer.callback_metrics[m_name]
                    if torch.is_tensor(metric):
                        metric = metric.item()
                    metric = round(metric, 3)
                    self.monitor_metrics[m_name].append(metric)
                else:
                    continue

    def get_avg_metric(self):
        list_avg = [self.config.dataset_name[0]]
        for i in range(1, len(self.stats_table.field_names)):
            m_name = self.stats_table.field_names[i]
            metrics = np.array(self.monitor_metrics[m_name])
            mean = round(float(np.mean(metrics)), 3)
            std = round(float(np.std(metrics)), 3)
            text = f"{mean} +- {std}"
            list_avg.append(text)
        self.stats_table.add_row(list_avg)
        return self.stats_table
