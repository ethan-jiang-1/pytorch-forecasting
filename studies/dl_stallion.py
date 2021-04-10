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
from studies.ds_stallion import DataSrc

class DataLoader(object):
    c_max_encoder_length = 36
    c_max_prediction_length = 6
    c_batch_size = 64

    @classmethod
    def get_max_encoder_length(cls):
        return cls.c_max_encoder_length

    @classmethod
    def set_max_encoder_length(cls, max_encoder_length):
        cls.c_max_encoder_length = max_encoder_length

    @classmethod
    def get_max_prediction_length(cls):
        return cls.c_max_prediction_length

    @classmethod
    def set_max_prediction_length(cls, max_prediction_length):
        cls.c_max_prediction_length = max_prediction_length

    @classmethod
    def get_batch_size(cls):
        return cls.c_batch_size

    @classmethod
    def set_batch_size(cls, batch_size):
        cls.c_batch_size = batch_size

    @classmethod
    def get_training_dataset(cls, data):
        special_days = DataSrc.get_special_days()

        training_cutoff = data["time_idx"].max() - cls.get_max_prediction_length()
        max_encoder_length = cls.get_max_encoder_length()
        max_prediction_length = cls.get_max_prediction_length()

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
    def get_dataloaders(cls, training, validation):
        #validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        batch_size = cls.get_batch_size()
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        return train_dataloader, val_dataloader

    @classmethod
    def save_datasets(cls, data):
        training = cls.get_training_dataset(data)
        validation = cls.get_validation_dataset(training, data)

        # save datasets
        training.save("training.pkl")
        validation.save("validation.pkl")


class DataExplorer(object):
    @classmethod
    def explore_dataset(cls, dataset, name):
        print("")
        print("inspect_dataloader", name)
        parameters = dataset.get_parameters()
        print("dataset {} parameters".format(name))
        for key in parameters.keys():
            print("  ", key, "=", parameters[key], " / ", type(parameters[key]))
        print("")

    @classmethod
    def explore_dataloader(cls, dataloader, name):
        writer = SummaryWriter()

        count = 0
        examples = 0
        print("")
        print("inspect_dataloader", name)
    
        for x0, x1 in iter(dataloader):
            if count == 0:
                print("encoder_cat.shape", x0["encoder_cat"].shape)
                print("encoder_cont.shape", x0["encoder_cont"].shape)
                print("encoder_target.shape", x0["encoder_target"].shape)
                print("encoder_lengths.shape", x0["encoder_lengths"].shape)
                print("")
                print("decoder_cat.shape", x0["decoder_cat"].shape)
                print("decoder_cont.shape", x0["decoder_cont"].shape)
                print("decoder_target.shape", x0["decoder_target"].shape)
                print("decoder_lengths.shape", x0["decoder_lengths"].shape)
                print("")
                print("decoder_time_idx.shape", x0["decoder_time_idx"].shape)
                print("groups.shape", x0["groups"].shape)
                print("target_scale.shape", x0["target_scale"].shape)

                print("x1_0.shape", x1[0].shape)
                print("x1_1", None)

            count += 1
            examples += x1[0].shape[0]

        print("")
        print("Summarize the dataloader output nums:", name)
        print("total batches:", count)
        print("batch size:", DataLoader.get_batch_size())
        print("total examples:", examples)

        writer.close()


def main():
    data = DataSrc.get_df_data()
    print(data)
    print(data.describe())

    training = DataLoader.get_training_dataset(data)
    validation = DataLoader.get_validation_dataset(training, data)

    DataExplorer.explore_dataset(validation, "validation")

    train_dataloader, val_dataloader = DataLoader.get_dataloaders(training, validation)
    print(train_dataloader)
    print(val_dataloader)

    DataExplorer.explore_dataloader(val_dataloader, "val")


if __name__ == '__main__':
    main()
