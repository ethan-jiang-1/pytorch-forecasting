#from pathlib import Path
#import pickle
from os import uname
import warnings

import numpy as np
#import pandas as pd
from pandas.core.common import SettingWithCopyWarning
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#import torch

#from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
#from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data
from pytorch_forecasting.data.examples import get_stallion_data
#from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
#from pytorch_forecasting.utils import profile

#import warnings
warnings.simplefilter("error", category=SettingWithCopyWarning)
warnings.simplefilter("ignore", category=FutureWarning)

class StlDataSrc(object):
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
    def _get_df_data_from_src(cls):
        """
        Demand data with covariates.

        ~20k samples of 350 timeseries. Important columns

        * Timeseries can be identified by ``agency`` and ``sku``.
        * ``volume`` is the demand
        * ``date`` is the month of the demand.

        Returns:
            pd.DataFrame: data
        """
        return get_stallion_data().copy()

    @classmethod
    def get_df_data(cls):
        data = cls._get_df_data_from_src()

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


def profiling_dataframe_to_html(data):
    from studies.ce_datasrc import DataSrcExplorer
    return DataSrcExplorer.profiling_dataframe_to_html(data, "df_profiing_stallion.html")

def _explore_sd(df_sd):
    from studies.ce_datasrc import DataSrcExplorer
    return DataSrcExplorer.explore_single_df(df_sd)


def explore_date(data):
    print("##explore_date")
    df_sd = data["date"]
    desp, unique_vals = _explore_sd(df_sd)
    print("unique date no is", desp[1], " out of raw:", desp[0])   
    print("unique vals:", unique_vals) 
    print("")

def explore_sku(data):
    print("##explore_sku")
    df_sd = data["sku"]
    desp, unique_vals = _explore_sd(df_sd)
    print("unique sku no is", desp[1], " out of raw:", desp[0])    
    print("unique vals", unique_vals)
    print("") 

def explore_agency(data):
    print("##explore_date")
    df_sd = data["agency"]
    desp, unique_vals = _explore_sd(df_sd)
    print("unique agency no is", desp[1], " out of raw:", desp[0])    
    print("unique vals", unique_vals) 
    print("")

def explore_time_idx(data):
    print("##explore_time_idx")
    print("time_idx.max", data["time_idx"].max())
    print("time_idx.max", data["time_idx"].min())
    df_data_time_idx_50 = data[lambda x: x.time_idx == 50]
    print(df_data_time_idx_50)
    print(df_data_time_idx_50.describe())
    print("")

def main():
    data = StlDataSrc.get_df_data()
    print(data)
    print(data.describe())

    print("columns:", len(data.columns))
    for cl in data.columns:
        print("  ", cl)

    explore_time_idx(data)
    explore_date(data)
    explore_sku(data)
    explore_agency(data)

    profiling_dataframe_to_html(data)


if __name__ == '__main__':
    main()
