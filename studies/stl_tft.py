#from pathlib import Path
import pickle
from pytorch_forecasting.models.nn import embeddings
#import warnings

#import numpy as np
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
#import torch

from pytorch_forecasting import TemporalFusionTransformer 
import pytorch_forecasting
#from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.callbacks.base import Callback
import os

#from pytorch_forecasting.utils import profile


class MyMetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class MyTbLogger(TensorBoardLogger):
    def __init__(self,         
                save_dir: str,
                name = "default",
                version = None,
                log_graph = False,
                default_hp_metric =True,
                prefix = '',
                **kwargs):
        super(MyTbLogger, self).__init__(save_dir, name=name, version=version, log_graph=log_graph, default_hp_metric=default_hp_metric, prefix=prefix, **kwargs)

    def log_hyperparams(self, params, metrics=None):
        return super(MyTbLogger, self).log_hyperparams(params, metrics=metrics)

    def log_metrics(self, metrics, step=None):
        if 'val_loss' in metrics:
            print(' val_loss:{:.2f}'.format(metrics['val_loss']))
        if 'loss' in metrics:
            print(" loss:{:.2f}".format(metrics['loss']))
        return super(MyTbLogger, self).log_metrics(metrics, step=step)


class StlTftExec(object):
    mcb = None
    log_dir = None

    @classmethod 
    def get_tb_log_dir(cls):
        if cls.log_dir is None:
            file_dir = os.path.dirname(__file__)
            app_dir = os.path.dirname(file_dir)
            cls.log_dir = app_dir + "/runs"
            if not os.path.isdir(cls.log_dir):
                os.makedirs(cls.log_dir, exist_ok=True)
            print("**tb_log_dir location:", cls.log_dir)
        return cls.log_dir

    @classmethod
    def get_tb_logger(cls):
        log_dir = cls.get_tb_log_dir()
        tb_logger = MyTbLogger(log_dir)
        return tb_logger

    @classmethod
    def get_metrics_monitor(cls):
        if cls.mcb is not None:
            return cls.mcb
        cls.mcb = MyMetricsCallback()
        return cls.mcb

    @classmethod
    def get_resume_from_checkpoint(cls):
        return None

    @classmethod 
    def get_trainer(cls, hp, gpus=0, min_epochs=2, max_epochs=40, resume_from_checkpoint=None, root_dir=".", tb_logger=None):
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        my_monitor = cls.get_metrics_monitor()
        tb_logger = cls.get_tb_logger()

        trainer = pl.Trainer(
            gpus=gpus,

            min_epochs=min_epochs,
            max_epochs=max_epochs,  # 40, #ethan
            default_root_dir=root_dir,
            weights_save_path=root_dir,
            #profiler="advanced",
            resume_from_checkpoint=resume_from_checkpoint,
            logger=tb_logger,

            #flush_logs_every_n_steps=20,
            #progress_bar_refresh_rate=20,

            #hyperparameter
            gradient_clip_val=hp.gradient_clip_val,
            #weights_summary="top",
            #limit_train_batches=30,
            # val_check_interval=20,
            # limit_val_batches=1,
            # fast_dev_run=True,
            # logger=logger,
            # profiler=True,
            callbacks=[lr_logger, early_stop_callback, my_monitor],
        )
        return trainer

    @classmethod
    def get_tft_model(cls, hp, training,):
        qtloss = QuantileLoss()
        tft = TemporalFusionTransformer.from_dataset(
            training,

            learning_rate=hp.learning_rate,
            hidden_size=hp.hidden_size,
            attention_head_size=hp.attention_head_size,
            dropout=hp.dropout,
            hidden_continuous_size=hp.hidden_continuous_size,

            output_size=7,  # 7 quantiles by default ?
            loss=qtloss,
            log_interval=10,
            log_val_interval=1,
            reduce_on_plateau_patience=3,)
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
            tft, 
            train_dataloader=train_dataloader, 
            val_dataloaders=val_dataloader, 
            min_lr=1e-5, 
            max_lr=1e2)
        print(f"suggested learning rate: {res.suggestion()}")
        # tft.hparams.learning_rate = res.suggestion()

        if plot_res:
            fig = res.plot(show=True, suggest=True)
            fig.show()
        return res.suggestion()

    @classmethod
    def train(cls, trainer, tft, train_dataloader, val_dataloader):
        print("Trainer.max_epochs", trainer.max_epochs)
        print("Trainer.min_epochs", trainer.min_epochs)
        ret = trainer.fit(
            tft,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        return ret

    @classmethod
    def predict(cls, tft, val_dataloader):
        # make a prediction on entire validation set
        preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)
        return preds, index


def main():

    from studies.stl_datasrc import StlDataSrc
    from studies.stl_dataloader import StlDataLoader
    from studies.ce_tft import TftExplorer
    from studies.ce_hyperparameters import HyperParameters

    hp = HyperParameters()
    print(hp)

    print(pytorch_forecasting.__version__)

    data = StlDataSrc.get_df_data()
    print(data)

    training = StlDataLoader.get_training_dataset(hp, data)
    validation = StlDataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = StlDataLoader.get_dataloaders(hp, training, validation)

    trainer = StlTftExec.get_trainer(hp, max_epochs=2)
    tft = StlTftExec.get_tft_model(hp, training)

    sum = ModelSummary(tft)
    print("\ntft.summary")
    print(sum)
    print("\ntft.hparams")
    print(tft.hparams)

    if hasattr(training, "hack_from_dataset_new_kwargs"):
        opt_explore_tft_from_dataset = False
        if opt_explore_tft_from_dataset:
            if hasattr(training, "hack_from_dataset_new_kwargs"):
                new_kwargs = training.hack_from_dataset_new_kwargs
                TftExplorer.explore_tft_from_dataset_new_kwargs(new_kwargs)

        opt_explore_tft_inputs = False
        if opt_explore_tft_inputs:
            TftExplorer.explore_tft_inputs(training, train_dataloader)

    opt_find_init_lr = False
    if opt_find_init_lr:
        trainer_lr = StlTftExec.get_trainer(hp, max_epochs=1)
        StlTftExec.find_init_lr(trainer_lr, tft, train_dataloader, val_dataloader)

    opt_exec_train_and_pred = True
    if opt_exec_train_and_pred:
        StlTftExec.train(trainer, tft, train_dataloader, val_dataloader)

    preds, index = StlTftExec.predict(tft, val_dataloader)
    print(preds)
    print(index)


if __name__ == '__main__':
    main()

