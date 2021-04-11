#from pathlib import Path
import pickle
from pytorch_forecasting.models.nn import embeddings
#import warnings

#import numpy as np
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#import torch

from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
import pytorch_forecasting
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
        qtloss = QuantileLoss()
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
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
            tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
        )
        print(f"suggested learning rate: {res.suggestion()}")
        tft.hparams.learning_rate = res.suggestion()

        if plot_res:
            fig = res.plot(show=True, suggest=True)
            fig.show()
        return res.suggestion()

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

class TftExplorer(object):
    @classmethod
    def explore_tft_from_dataset_new_kwargs(cls, new_kwargs):
        print("##TftExplorer.explore_tft_from_dataset_new_kwargs")
        for key in new_kwargs.keys():
            print("  ", key, "\t", new_kwargs[key])
        print("")
        print("key vals:")
        embedding_labels = new_kwargs["embedding_labels"]
        print("embedding_labels", len(embedding_labels.keys()))
        for i, key in enumerate(embedding_labels.keys()):
            print("  [{}]".format(i), key, "", embedding_labels[key])
        print("")
        print("time_varying_reals_encoder", len(new_kwargs["time_varying_reals_encoder"]))
        print(" ", new_kwargs["time_varying_reals_encoder"])

        x_reals = new_kwargs["x_reals"]
        print("x_reals", len(x_reals))
        for key in x_reals:
            print("  ", key)

        x_categoricals = new_kwargs["x_categoricals"]
        print("x_reals", len(x_categoricals))
        for key in x_categoricals:
            print("  ", key)    

        print("")

    @classmethod
    def explore_tft_inputs(cls, dataset, dataloader):
        from studies.dl_stallion import DataExplorer
        if hasattr(dataset, "hack_from_dataset_new_kwargs"):
            print("##TftExplorer.explore_tft_inputs")
            new_kwargs = dataset.hack_from_dataset_new_kwargs    

            x_categoricals = new_kwargs["x_categoricals"]
            print("x_categoricals (encoder_cat?)", len(x_categoricals))
            for i, key in enumerate(x_categoricals):
                print("  ", i, key)
            print("")

            time_varying_reals_encoder = new_kwargs["time_varying_reals_encoder"]  
            x_reals = new_kwargs["x_reals"]
            print("x_reals (encoder_cont?)", len(x_reals))
            for i, key in enumerate(x_reals):
                unknown = False
                if key in time_varying_reals_encoder:
                    unknown = True
                if unknown:
                    print("  ?",i, key)                
                else:
                    print("  !",i, key)
            print("") 

            time_varying_reals_encoder = new_kwargs["time_varying_reals_encoder"]  
            print("time_varying_reals_encoder", len(time_varying_reals_encoder))
            for key in time_varying_reals_encoder:
                print("  ", key)
            print("")

            time_varying_reals_decoder = new_kwargs["time_varying_reals_decoder"]  
            print("time_varying_reals_decoder", len(time_varying_reals_decoder))
            for key in time_varying_reals_decoder:
                print("  ", key)
            print("")

            DataExplorer.explore_dataloader(dataloader, "train")
    
def main():

    from studies.ds_stallion import DataSrc
    from studies.dl_stallion import DataLoader
    from studies.tft_model import TftExec

    print(pytorch_forecasting.__version__)

    data = DataSrc.get_df_data()
    print(data)

    training = DataLoader.get_training_dataset(data)
    validation = DataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = DataLoader.get_dataloaders(training, validation)

    trainer = TftExec.get_trainer(max_epochs=3)
    tft = TftExec.get_tft_model(training)
    if hasattr(training, "hack_from_dataset_new_kwargs"):
        #new_kwargs = training.hack_from_dataset_new_kwargs
        #TftExplorer.explore_tft_from_dataset_new_kwargs(new_kwargs)

        TftExplorer.explore_tft_inputs(training, train_dataloader)

    exec_train_and_pred = False
    if exec_train_and_pred:
        TftExec.find_init_lr(trainer, tft, train_dataloader, val_dataloader)
        TftExec.train(trainer, tft, train_dataloader, val_dataloader)
        study = TftExec.turn_hyperparameters(train_dataloader, val_dataloader, n_trials=2, max_epochs=2)
        print(study)

        preds, index = TftExec.predict(tft, val_dataloader)
        print(preds)
        print(index)


if __name__ == '__main__':
    main()

