#from pathlib import Path
#import pickle
import warnings

import numpy as np
#import pandas as pd
from pandas.core.common import SettingWithCopyWarning
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#import torch

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data
#from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
#from pytorch_forecasting.utils import profile

#import warnings
warnings.simplefilter("error", category=SettingWithCopyWarning)

class DataSrc(object):
    @classmethod
    def get_special_days(cls):
        special_days = [
            "easter_day",
            "good_friday",
            "new_year",
            "christmas",
            "labor_day",
            "independence_day",
            "revolution_day_memorial",
            "regional_games",
            "fifa_u_17_world_cup",
            "football_gold_cup",
            "beer_capital",
            "music_fest",
        ]
        return special_days

    @classmethod
    def get_df_data(cls):
        data = get_stallion_data().copy()

        data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
        data["time_idx"] -= data["time_idx"].min()

        data["month"] = data.date.dt.month.astype("str").astype("category")  # categories have be strings
        data["log_volume"] = np.log(data.volume + 1e-8)

        data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
        data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

        # data = data[lambda x: (x.sku == data.iloc[0]["sku"]) & (x.agency == data.iloc[0]["agency"])]

        special_days = cls.get_special_days()
        data[special_days] = data[special_days].apply(lambda x: x.map({0: "", 1: x.name})).astype("category")
        return data


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

def profiling_dataframe(df):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title="Pandas Profiling Report")
    return profile

def main():
    data = DataSrc.get_df_data()
    print(data)
    print(data.describe())

    generate_profiling = True
    if generate_profiling:
        profile = profiling_dataframe(data)
        print(profile)
        profile.to_file("df_profiling_stallion.html")

    training = DataLoader.get_training_dataset(data)
    validation = DataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = DataLoader.get_dataloaders(training, validation)
    print(train_dataloader)
    print(val_dataloader)


if __name__ == '__main__':
    main()
