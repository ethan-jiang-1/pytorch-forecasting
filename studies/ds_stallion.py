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


if __name__ == '__main__':
    main()
