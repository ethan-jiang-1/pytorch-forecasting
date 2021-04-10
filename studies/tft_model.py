#from pathlib import Path
import pickle
#import warnings

#import numpy as np
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#import torch

from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile


class TftExec(object):
    @classmethod 
    def get_trainer(cls, max_epochs=100):
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()

        xtrainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            weights_summary="top",
            gradient_clip_val=0.1,
            limit_train_batches=30,
            # val_check_interval=20,
            # limit_val_batches=1,
            # fast_dev_run=True,
            # logger=logger,
            # profiler=True,
            callbacks=[lr_logger, early_stop_callback],
        )
        return xtrainer

    @classmethod
    def get_tft_model(cls, training):
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            log_val_interval=1,
            reduce_on_plateau_patience=3,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        return tft

    @classmethod
    def find_init_lr(cls, trainer, tft, train_dataloader, val_dataloader, plot_res=False):
        # find optimal learning rate
        # remove logging and artificial epoch size
        tft.hparams.log_interval = -1
        tft.hparams.log_val_interval = -1
        trainer.limit_train_batches = 1.0
        # run learning rate finder
        res = trainer.tuner.lr_find(
            tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
        )
        print(f"suggested learning rate: {res.suggestion()}")
        tft.hparams.learning_rate = res.suggestion()

        if plot_res:
            fig = res.plot(show=True, suggest=True)
            fig.show()

    @classmethod
    def train(cls, trainer, tft, train_dataloader, val_dataloader):
        trainer.fit(
            tft,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    @classmethod
    def predict(cls, tft, val_dataloader):
        # make a prediction on entire validation set
        preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)
        return preds, index

    @classmethod
    def turn_hyperparameters(cls, train_dataloader, val_dataloader, n_trials=200, max_epochs=50):
        # tune
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path="optuna_test",
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )

        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)
        return study

class TftProfile(object):
    @classmethod
    def profile_training(cls, trainer, tft, train_dataloader, val_dataloader):
        #profile speed
        profile(
            trainer.fit,
            profile_fname="profile.prof",
            model=tft,
            period=0.001,
            filter="pytorch_forecasting",
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
