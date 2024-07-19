from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.nn import aggr, SchNet

from conan_fgw.src.model.graph_embeddings.esan import (
    AverageConformerESAN,
    GeometryInducedESAN,
    Geometry2DInducedESAN,
)
from conan_fgw.src.model.graph_embeddings.gat import GATBased
from conan_fgw.src.model.graph_embeddings.schnet_no_sum import (
    SchNetNoSum,
    SchNetWithMultipleReturns,
)
from conan_fgw.src.model.graph_embeddings.dimenet import DimeNet
from conan_fgw.src.model.graph_embeddings.visnet import ViSNet
from conan_fgw.src.trainer import TrainerHolder
import time


class MoleculeNetClassificationModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        learning_rate: float,
        class_weights=None,
        is_distributed: bool = False,
        trade_off: bool = False,
    ):
        super().__init__()
        self.batch_size = None
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.is_distributed = is_distributed

        ## handling the AUC-ROC
        self.train_y_pred = []
        self.train_y_true = []

        self.val_y_pred = []
        self.val_y_true = []

        self.test_y_pred = []
        self.test_y_true = []

        self.trade_off = trade_off

        self.start_time = time.time()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, verbose=True
        )  # factor = 0.5
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }  # val_loss

    def training_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        self.train_y_pred.append(y_pred)
        self.train_y_true.append(y_true)
        self._log_metrics({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        self.val_y_pred.append(y_pred)
        self.val_y_true.append(y_true)
        self._log_metrics({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        self.test_y_pred.append(y_pred)
        self.test_y_true.append(y_true)
        self._log_metrics({"test_loss": loss})
        return loss

    def _shared_step(self, batch):
        y_pred, y_true = self.get_predictions(batch)
        loss = MoleculeNetClassificationModel.classification_loss(
            y_pred, y_true, self.class_weights
        )
        return y_pred.detach().cpu(), y_true.detach().cpu(), loss

    def shared_epoch_end(self, outputs, stage):
        y_pred, y_true = outputs
        metric = TrainerHolder.classification_metric(y_pred, y_true, self.trade_off)
        return metric

    def on_train_epoch_end(self):
        self.train_y_pred = torch.cat(self.train_y_pred, dim=0)
        self.train_y_true = torch.cat(self.train_y_true, dim=0)
        metric = self.shared_epoch_end((self.train_y_pred, self.train_y_true), "train")
        self._log_metrics(
            {
                f"train_{TrainerHolder.classification_metric_name(self.trade_off)[i]}": metric[
                    i
                ].item()
                for i in range(len(metric))
            },
            sync_dist=True,
            rank_zero_only=False,
        )
        self._log_metrics({"runtime": time.time() - self.start_time})
        self.train_y_pred = []
        self.train_y_true = []

    def on_validation_epoch_end(self):
        if self.is_distributed:
            val_y_pred_comb = torch.cat(self.val_y_pred, dim=0)
            val_y_true_comb = torch.cat(self.val_y_true, dim=0)
            val_y_pred_comb = self.all_gather(val_y_pred_comb)
            val_y_true_comb = self.all_gather(val_y_true_comb)
            # reshape to (dataset_size, *other_dims)
            world_size = torch.cuda.device_count()
            new_batch_size = world_size * val_y_pred_comb.shape[1]
            val_y_pred_comb = val_y_pred_comb.view(new_batch_size, *val_y_pred_comb.shape[2:])
            val_y_true_comb = val_y_true_comb.view(new_batch_size, *val_y_true_comb.shape[2:])

            # if self.trainer.is_global_zero:
            metric = self.shared_epoch_end((val_y_pred_comb, val_y_true_comb), "val")
            self._log_metrics(
                {
                    f"val_{TrainerHolder.classification_metric_name(self.trade_off)[i]}": metric[
                        i
                    ].item()
                    for i in range(len(metric))
                },
                sync_dist=False,
                rank_zero_only=True,
            )
            # else:
            #     num_metrics = 3 if self.trade_off else 2
            #     self.trainer.callback_metrics.update(
            #         {f'val_{TrainerHolder.classification_metric_name(self.trade_off)[i]}': torch.tensor(0.) for i in range(num_metrics)}
            #     )
            self.val_y_pred.clear()
            self.val_y_true.clear()
        else:
            self.val_y_pred = torch.cat(self.val_y_pred, dim=0)
            self.val_y_true = torch.cat(self.val_y_true, dim=0)
            metric = self.shared_epoch_end((self.val_y_pred, self.val_y_true), "val")
            self._log_metrics(
                {
                    f"val_{TrainerHolder.classification_metric_name(self.trade_off)[i]}": metric[
                        i
                    ].item()
                    for i in range(len(metric))
                }
            )
            self.val_y_pred = []
            self.val_y_true = []

    def on_test_epoch_end(self):
        if self.is_distributed:
            test_y_pred_comb = torch.cat(self.test_y_pred, dim=0)
            test_y_true_comb = torch.cat(self.test_y_true, dim=0)
            test_y_pred_comb = self.all_gather(test_y_pred_comb)
            test_y_true_comb = self.all_gather(test_y_true_comb)
            # reshape to (dataset_size, *other_dims)
            world_size = torch.cuda.device_count()
            new_batch_size = world_size * test_y_pred_comb.shape[1]
            test_y_pred_comb = test_y_pred_comb.view(new_batch_size, *test_y_pred_comb.shape[2:])
            test_y_true_comb = test_y_true_comb.view(new_batch_size, *test_y_true_comb.shape[2:])

            # if self.trainer.is_global_zero:
            metric = self.shared_epoch_end((test_y_pred_comb, test_y_true_comb), "test")
            self._log_metrics(
                {
                    f"test_{TrainerHolder.classification_metric_name(self.trade_off)[i]}": metric[
                        i
                    ].item()
                    for i in range(len(metric))
                },
                sync_dist=False,
                rank_zero_only=True,
            )
            self.test_y_pred.clear()
            self.test_y_true.clear()
        else:
            self.test_y_pred = torch.cat(self.test_y_pred, dim=0)
            self.test_y_true = torch.cat(self.test_y_true, dim=0)
            metric = self.shared_epoch_end((self.test_y_pred, self.test_y_true), "test")
            self._log_metrics(
                {
                    f"test_{TrainerHolder.classification_metric_name(self.trade_off)[i]}": metric[
                        i
                    ].item()
                    for i in range(len(metric))
                }
            )
            self.test_y_pred = []
            self.test_y_true = []

    @abstractmethod
    def get_predictions(self, batch):
        pass

    @staticmethod
    def classification_loss(predicted, expected, class_weights=None):
        # return F.cross_entropy(predicted, expected)
        if class_weights:
            if expected.get_device() >= 0:
                class_weights = class_weights.to(expected.get_device())
        loss = F.binary_cross_entropy(predicted, expected, weight=class_weights)
        return loss

    def _log_metrics(
        self, metrics, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=False
    ):
        """
        on_step=False, on_epoch=True, sync_dist=True
        """
        self.log_dict(
            metrics,
            batch_size=self.batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )


class MoleculeNetRegressionModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, learning_rate: float, is_distributed: bool = False):
        super().__init__()
        self.batch_size = None
        self.learning_rate = learning_rate
        self.is_distributed = is_distributed

        ## handling the multi-GPUs
        self.train_y_pred = []
        self.train_y_true = []

        self.val_y_pred = []
        self.val_y_true = []

        self.test_y_pred = []
        self.test_y_true = []
        self.start_time = time.time()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.8, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.train_y_pred.append(y_pred)
        self.train_y_true.append(y_true)
        self._log_metrics({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        self.val_y_pred.append(y_pred)
        self.val_y_true.append(y_true)
        self._log_metrics({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_true, loss = self._shared_step(batch)
        self.test_y_pred.append(y_pred)
        self.test_y_true.append(y_true)
        self._log_metrics({"test_loss": loss})
        return loss

    def _shared_step(self, batch):
        y_pred, y_true = self.get_predictions(batch)
        loss = MoleculeNetRegressionModel.regression_loss(y_pred, y_true)
        return y_pred.detach().cpu(), y_true.detach().cpu(), loss

    def shared_epoch_end(self, outputs, stage):
        y_pred, y_true = outputs
        metric = TrainerHolder.regression_metric(y_pred, y_true)
        return metric

    def on_train_epoch_end(self):
        self.train_y_pred = torch.cat(self.train_y_pred, dim=0)
        self.train_y_true = torch.cat(self.train_y_true, dim=0)
        metric = self.shared_epoch_end((self.train_y_pred, self.train_y_true), "train")
        self._log_metrics(
            {f"train_{TrainerHolder.regression_metric_name()}": metric.item()}, rank_zero_only=True
        )
        self._log_metrics({"runtime": time.time() - self.start_time})
        self.train_y_pred = []
        self.train_y_true = []

    def on_validation_epoch_end(self):
        if self.is_distributed:
            val_y_pred_comb = torch.cat(self.val_y_pred, dim=0)
            val_y_true_comb = torch.cat(self.val_y_true, dim=0)
            val_y_pred_comb = self.all_gather(val_y_pred_comb)
            val_y_true_comb = self.all_gather(val_y_true_comb)
            # reshape to (dataset_size, *other_dims)
            world_size = torch.cuda.device_count()
            new_batch_size = world_size * val_y_pred_comb.shape[1]
            val_y_pred_comb = val_y_pred_comb.view(new_batch_size, *val_y_pred_comb.shape[2:])
            val_y_true_comb = val_y_true_comb.view(new_batch_size, *val_y_true_comb.shape[2:])

            metric = self.shared_epoch_end((val_y_pred_comb, val_y_true_comb), "val")
            self._log_metrics(
                {f"val_{TrainerHolder.regression_metric_name()}": metric.item()},
                sync_dist=False,
                rank_zero_only=True,
            )
            self.val_y_pred.clear()
            self.val_y_true.clear()
        else:
            self.val_y_pred = torch.cat(self.val_y_pred, dim=0)
            self.val_y_true = torch.cat(self.val_y_true, dim=0)
            metric = self.shared_epoch_end((self.val_y_pred, self.val_y_true), "val")
            self._log_metrics({f"val_{TrainerHolder.regression_metric_name()}": metric.item()})
            self.val_y_pred = []
            self.val_y_true = []

    def on_test_epoch_end(self):
        if self.is_distributed:
            test_y_pred_comb = torch.cat(self.test_y_pred, dim=0)
            test_y_true_comb = torch.cat(self.test_y_true, dim=0)
            test_y_pred_comb = self.all_gather(test_y_pred_comb)
            test_y_true_comb = self.all_gather(test_y_true_comb)
            # reshape to (dataset_size, *other_dims)
            world_size = torch.cuda.device_count()
            new_batch_size = world_size * test_y_pred_comb.shape[1]
            test_y_pred_comb = test_y_pred_comb.view(new_batch_size, *test_y_pred_comb.shape[2:])
            test_y_true_comb = test_y_true_comb.view(new_batch_size, *test_y_true_comb.shape[2:])

            # if self.trainer.is_global_zero:
            metric = self.shared_epoch_end((test_y_pred_comb, test_y_true_comb), "test")
            self._log_metrics(
                {f"test_{TrainerHolder.regression_metric_name()}": metric.item()},
                sync_dist=False,
                rank_zero_only=True,
            )
            self.test_y_pred.clear()
            self.test_y_true.clear()
        else:
            self.test_y_pred = torch.cat(self.test_y_pred, dim=0)
            self.test_y_true = torch.cat(self.test_y_true, dim=0)
            metric = self.shared_epoch_end((self.test_y_pred, self.test_y_true), "test")
            self._log_metrics({f"test_{TrainerHolder.regression_metric_name()}": metric.item()})
            self.test_y_pred = []
            self.test_y_true = []

    @abstractmethod
    def get_predictions(self, batch):
        pass

    @staticmethod
    def regression_loss(predicted, expected):
        return F.mse_loss(predicted, expected)

    def _log_metrics(
        self, metrics, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=False
    ):
        """
        on_step=False, on_epoch=True, sync_dist=True
        """
        self.log_dict(
            metrics,
            batch_size=self.batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )


class EquivAggregation(MoleculeNetRegressionModel, metaclass=ABCMeta):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name: str,
        is_distributed: bool = False,
    ):
        super().__init__(learning_rate=learning_rate, is_distributed=is_distributed)

        self.num_conformers = num_conformers
        self.node_embeddings_model = EquivModelsHolder.get_model(
            model_name, self.device, feat_dim=128
        )
        self.batch_size = batch_size
        self.conformers_mean_aggr = aggr.MeanAggregation()

    def get_predictions(self, batch):
        batch, node_index = batch
        conformers_index = self.create_aggregation_index(batch)
        y_pred = self(batch, conformers_index, node_index)
        y_true = self.conformers_mean_aggr(batch.y.view(-1, 1), conformers_index)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        return y_pred, y_true

    def create_aggregation_index(self, batch):
        index = []
        mol_idx = -1
        i = 0
        while i < len(batch.smiles):
            mol_idx += 1
            for _ in range(self.num_conformers):
                index.append(mol_idx)
                i += 1
        return torch.Tensor(index).long().to(self.device)


class EquivAggregationClassification(MoleculeNetClassificationModel, metaclass=ABCMeta):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name: str,
        class_weights=None,
        is_distributed: bool = False,
        trade_off: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            class_weights=class_weights,
            is_distributed=is_distributed,
            trade_off=trade_off,
        )
        self.num_conformers = num_conformers
        self.node_embeddings_model = EquivModelsHolder.get_model(
            model_name, self.device, feat_dim=512, cutoff=10.0
        )
        self.batch_size = batch_size
        self.conformers_mean_aggr = aggr.MeanAggregation()

    def get_predictions(self, batch):
        batch, node_index = batch
        conformers_index = self.create_aggregation_index(batch)
        y_pred = self(batch, conformers_index, node_index)
        y_true = self.conformers_mean_aggr(batch.y.view(-1, 1), conformers_index)
        return y_pred, y_true

    def create_aggregation_index(self, batch):
        index = []
        mol_idx = -1
        i = 0
        while i < len(batch.smiles):
            mol_idx += 1
            for _ in range(self.num_conformers):
                index.append(mol_idx)
                i += 1
        return torch.Tensor(index).long().to(self.device)


class EquivModelsHolder:
    @staticmethod
    def get_model(name: str, device, **kwargs):
        if name == "simple_dimenet":
            return DimeNet(
                # Default parameters from pytorch geometric examples (pretrained on QM9 version)
                device,
                hidden_channels=3,
                out_channels=1,
                num_blocks=1,
                num_bilinear=1,
                num_spherical=2,
                num_radial=1,
                cutoff=5.0,
                envelope_exponent=1,
                num_before_skip=1,
                num_after_skip=1,
                num_output_layers=1,
            )
        elif name == "dimenet":
            model = DimeNet(
                # Default parameters from pytorch geometric examples (pretrained on QM9 version)
                device,
                hidden_channels=kwargs.get("feat_dim"),
                out_channels=kwargs.get("feat_dim") // 2,
                num_blocks=6,
                num_bilinear=8,
                num_spherical=2,
                num_radial=3,
                cutoff=5.0,
                envelope_exponent=5,
                num_before_skip=1,
                num_after_skip=2,
                num_output_layers=3,
                use_covalent=False,
                num_interactions=6,  ## add-on
                num_gaussians=10,  ## add-on
                num_filters=64,  ## add-on
            )
            model.hidden_channels = kwargs.get("feat_dim")
            return model
        elif name == "simple_schnet":
            return SchNet()
        elif name == "schnet":
            if "cutoff" in kwargs:
                return SchNetNoSum(
                    device,
                    hidden_channels=kwargs.get("feat_dim"),
                    use_covalent=False,
                    cutoff=kwargs.get("cutoff"),
                    num_gaussians=10,
                    num_filters=256,
                    num_interactions=3,
                )
            else:
                return SchNetNoSum(
                    device,
                    hidden_channels=kwargs.get("feat_dim"),
                    use_covalent=False,
                    num_interactions=3,
                )
        elif name == "schnet_covalent":
            return SchNetNoSum(device, use_covalent=True)
        elif name == "avg_conf_esan":
            return AverageConformerESAN(device)
        elif name == "gat":
            return GATBased(
                out_channels=kwargs.get("feat_dim") // 2,
            )
        elif name == "geometry_induced_esan":
            return GeometryInducedESAN(device)
        elif name == "geometry_2d_induced_esan":
            return Geometry2DInducedESAN(device)
        elif name == "visnet":
            return ViSNet(
                device,
                hidden_channels=kwargs.get("feat_dim"),
            )
