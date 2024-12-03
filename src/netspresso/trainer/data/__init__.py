from netspresso.trainer.data.data import (
    DatasetConfig,
    ImageLabelPathConfig,
    LocalClassificationDatasetConfig,
    LocalDetectionDatasetConfig,
    LocalSegmentationDatasetConfig,
    PathConfig,
)

DATA_CONFIG_TYPE = {
    "classification": LocalClassificationDatasetConfig,
    "detection": LocalDetectionDatasetConfig,
    "segmentation": LocalSegmentationDatasetConfig,
}

__all__ = ["ImageLabelPathConfig", "PathConfig", "DATA_CONFIG_TYPE", "DatasetConfig"]
