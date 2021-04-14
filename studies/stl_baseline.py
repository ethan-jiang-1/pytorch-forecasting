#from pathlib import Path
#import pickle
#from pytorch_forecasting.models.nn import embeddings
#import warnings

#import numpy as np
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
import torch

#from pytorch_forecasting import TemporalFusionTransformer 
import pytorch_forecasting
#from pytorch_forecasting.data.examples import get_stallion_data
#from pytorch_forecasting.metrics import QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
#from pytorch_lightning.core.memory import ModelSummary
#from pytorch_lightning.callbacks.base import Callback
#from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting import Baseline


class StlTftBaseline(object):
    @classmethod
    def perform_baseline_mean(cls, val_dataloader):
        # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
        # actuals = torch.cat([y for x, y in iter(val_dataloader)])
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])  # ethan

        """
        Baseline model that uses last known target value to make prediction.
        """
        baseline_model = Baseline()

        baseline_predictions = baseline_model.predict(val_dataloader)

        print("")
        print("actuals.shape", actuals.shape)
        print("baseline_predictions.shape", baseline_predictions.shape)
        print("")

        baseline_ap_mean = (actuals - baseline_predictions).abs().mean().item()
        return baseline_ap_mean

def main():

    from studies.stl_datasrc import StlDataSrc
    from studies.stl_dataloader import StlDataLoader
    #from studies.ce_tft import TftExplorer
    from studies.ce_hyperparameters import HyperParameters
    from studies.ce_hyperparameters import DefaultHyperParametersStl

    opt_pred_3 = False
    if opt_pred_3:
        DefaultHyperParametersStl.MAX_ENCODER_LENGTH = 27
        DefaultHyperParametersStl.MAX_PREDICTION_LENGTH = 3

    hp = HyperParameters()
    print(hp)

    print(pytorch_forecasting.__version__)

    data = StlDataSrc.get_df_data()
    print(data)

    training = StlDataLoader.get_training_dataset(hp, data)
    validation = StlDataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = StlDataLoader.get_dataloaders(hp, training, validation)

    ap_mean = StlTftBaseline.perform_baseline_mean(val_dataloader)
    print("baseline: Baseline model that uses last known target value to make prediction.")
    print("baseline ap_mean (baseline)", ap_mean)


if __name__ == '__main__':
    main()

