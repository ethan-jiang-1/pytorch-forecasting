#from pathlib import Path
#import pickle
#import warnings

#import numpy as np
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#import torch

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
#from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data
#from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
#from pytorch_forecasting.utils import profile

#import warnings
#warnings.simplefilter("error", category=SettingWithCopyWarning)

from torch.utils.tensorboard import SummaryWriter
from studies.stl_datasrc import StlDataSrc

class StlDataLoader(object):
    @classmethod
    def get_training_dataset(cls, hp, data):
        special_days = StlDataSrc.get_special_days()

        training_cutoff = data["time_idx"].max() - hp.MAX_PREDICTION_LENGTH
        max_encoder_length = hp.MAX_ENCODER_LENGTH
        max_prediction_length = hp.MAX_PREDICTION_LENGTH

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],

            min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
            max_encoder_length=max_encoder_length,

            min_prediction_length=1,
            max_prediction_length=max_prediction_length,

            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            
            variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable

            time_varying_known_categoricals=["special_days", "month"],
            time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
                "avg_volume_by_agency",
                "avg_volume_by_sku",
            ],

            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            ),  # use softplus with beta=1.0 and normalize by group

            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        return training

    @classmethod
    def get_validation_dataset(cls, training, data):
        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        return validation

    @classmethod
    def get_dataloaders(cls, hp, training, validation):
        #validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        batch_size = hp.BATCH_SIZE
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        return train_dataloader, val_dataloader

    @classmethod
    def save_datasets(cls, hp, data):
        training = cls.get_training_dataset(hp, data)
        validation = cls.get_validation_dataset(training, hp, data)

        # save datasets
        training.save("training.pkl")
        validation.save("validation.pkl")


def main():
    from studies.ce_hyperparameters import HyperParameters
    from studies.ce_dataloader import DataLoaderExplorer

    hp = HyperParameters()

    data = StlDataSrc.get_df_data()
    print(data)
    print(data.describe())

    training = StlDataLoader.get_training_dataset(hp, data)
    validation = StlDataLoader.get_validation_dataset(training, data)

    DataLoaderExplorer.explore_dataset(validation, "validation")

    train_dataloader, val_dataloader = StlDataLoader.get_dataloaders(hp, training, validation)
    print(train_dataloader)
    print(val_dataloader)

    DataLoaderExplorer.explore_dataloader(val_dataloader, "val")


if __name__ == '__main__':
    main()
