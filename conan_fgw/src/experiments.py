from dataclasses import dataclass

from conan_fgw.src.data.datamodules import ConformersBasedDataModule
from conan_fgw.src.data.datasets import (
    LargeConformerBasedDataset,
    GEOMDataset,
    LargeConformerBasedDatasetNTrials,
)

from conan_fgw.src.model.schnet_based_models import (
    EmbeddingsWithGATAggregation,
    EmbeddingsWithGATAggregationClassification,
    EmbeddingsWithGAT,
    EmbeddingsWithGATAggregationBaryCenter,
    EmbeddingsWithGATAggregationClassificationBaryCenter,
    EmbeddingsVisualizationBaryCenter,
)


@dataclass
class SOTAExperiment:
    dataset_class = LargeConformerBasedDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregation


@dataclass
class SOTAExperimentBaryCenter:
    dataset_class = LargeConformerBasedDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregationBaryCenter


@dataclass
class SOTAClassificationExperiment:
    dataset_class = LargeConformerBasedDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregationClassification


@dataclass
class SOTAClassificationGEOMExperiment:
    dataset_class = GEOMDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregationClassification


@dataclass
class SOTAClassificationGEOMExperimentBaryCenter:
    dataset_class = GEOMDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregationClassificationBaryCenter


@dataclass
class SOTAClassificationExperimentBaryCenter:
    dataset_class = LargeConformerBasedDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregationClassificationBaryCenter


@dataclass
class TrialsExperiment:
    dataset_class = LargeConformerBasedDatasetNTrials
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregation


@dataclass
class DimeNetGEOMExperiment:
    dataset_class = GEOMDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGATAggregation


@dataclass
class GATExperiment:
    dataset_class = LargeConformerBasedDataset
    datamodule_class = ConformersBasedDataModule
    model_class = EmbeddingsWithGAT
