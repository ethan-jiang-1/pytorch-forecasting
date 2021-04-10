# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/phylypo/TimeSeriesPrediction/blob/main/Time_Series_DL_TFT_N_BEATS.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Overview of Time Series Forecasting from Statistical to Recent ML Approaches 
# 
# Topics for this notebook in bold:
# - Introduction to TS
# - Decompose (*Time_Series_FFT.ipynb*)
#  - Gen Synthic
#  - Decompose FFT
# - Naive approaches
# - Statistical (*Time_Series_ES_ARIMA.ipynb*)
#  - Smoothing techniques
#  - ARIMA
#  - State Space (*Time_Series_StateSpace.ipynb*)
# - ML (*Time_Series_ML-LR_XGBoost.ipynb*)
#   - Linear Regression
#   - Decision Tree (XGBoost)
# - DL (*Time_Series_DL_LSTM_CNN.ipynb*)
#  - LSTM, CNN + LSTM
#  - TCN (*Time_Series_DL_TCN_LSTNet.ipynb*)
#  - LSTNet
#  - **TFT (*Time_Series_DL_TFT_N-BEATS.ipynb*)**
#  - **N-BEATS**
# - Commercial: (*Time_Series_Commercial.ipynb*)
#  - Facebook Prophet
#  - Amazon DeepAR
# %% [markdown]
# ## Deep Learning
# %% [markdown]
# ## TFT
# Temporal Fusion Transformers (TFT)
# %% [markdown]
# 
# "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
# Bryan Lim, Sercan O. Arık, Nicolas Loeffb, Tomas Pfister, Dec 2019 ([Paper](https://arxiv.org/abs/1912.09363.pdf))
# 
# 
# ![multi-horizon](https://storage.googleapis.com/groundai-web-prod/media/users/user_236644/project_402585/images/Schematic.png)
# 
# 
# Compare to other models:
# ARIMA, ETS, TRMF, DeepAR, DSSM, ConvTrans, Seq2Seq, MQRNN
# - ETS (Error, Trend, Seasonal) method is an approach method for forecasting time series univariate. (https://otexts.com/fpp2/arima-ets.html)
# - Deep State-Space Models (DSSM) [6] adopt a similar approach, utilizing LSTMs to generate parameters of a predefined linear state-space model with predictive distributions produced via Kalman filtering – with extensions for multivariate time series data in [21].
# - Deep AR [9] which uses stacked LSTM layers to generate parameters of one-step-ahead Gaussian predictive distributions
# - The Multi-horizon Quantile Recurrent Forecaster (MQRNN) [10] uses LSTM or convolutional encoders to generate context vectors which are fed into multi-layer perceptrons (MLPs) for each horizon.
# 
# 
# In [11] a multi-modal attention mechanism is used with LSTM encoders to construct context vectors for a bi-directional LSTM decoder. Despite performing better than LSTM-based iterative methods, interpretability remains challenging for such standard direct methods.
# (C. Fan, et al., Multi-horizon time series forecasting with temporal attention learning, 2019.)
# 
# By interpreting attention patterns, TFT can provide insightful explanations about temporal dynamics, and do so while maintaining state-ofthe-art performance on a variety of datasets.
# 
# ![data](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/b941cb83f4aad597bf7ca72613d130c714d65d54/7-Table1-1.png)
# 
# Time Series Interpretability with Attention: 
# - Attention mechanisms are used in translation [17], image classification [22] or tabular learning [23]
# to identify salient portions of input for each instance using the magnitude of attention weights. 
# - Recently, they have been adapted for time series with interpretability motivations [7, 12, 24], using LSTM-based [25] and transformer-based [12] architectures. However, this was done without considering the importance
# of static covariates (as the above methods blend variables at each input). 
# - TFT alleviates this by using separate encoder-decoder attention for static features at each time step on top of the self-attention to determine the contribution time-varying inputs.
# 
# Ref:
# - https://arxiv.org/abs/1912.09363.pdf
# - https://github.com/google-research/google-research/tree/master/tft
# - https://github.com/louisyuzhe/deeplearning_forecast
# - https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
# %% [markdown]
# ## TFT Code
# %% [markdown]
# For this tutorial, we will use the Stallion dataset from Kaggle describing sales of various beverages. Our task is to make a six-month forecast of the sold volume by stock keeping units (SKU), that is products, sold by an agency, that is a store. There are about 21 000 monthly historic sales records. In addition to historic sales we have information about the sales price, the location of the agency, special days such as holidays, and volume sold in the entire industry.
# %% [markdown]
# ### Install libraries

# %%
#get_ipython().system('pip install pytorch_lightning')


# %%
#get_ipython().system('pip install pytorch_forecasting')


# %%
#get_ipython().system('pip list | grep torch')
#get_ipython().system('pip list | grep pytorch')


# %%
USING_LOCAL_PF = False
if USING_LOCAL_PF:
    get_ipython().system('git clone https://github.com/ethan-jiang-1/pytorch-forecasting.git pf')
    import os
    os.chdir("/content/pf")
    get_ipython().system('ls -l')


# %%
import pytorch_forecasting
print("pytorch_forecasting.__version__", pytorch_forecasting.__version__)
if pytorch_forecasting.__version__ == "0.0.0":
    print("using local version of pytorch_forecasting")


# %%
#https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import copy
import inspect
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer

from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


# %%
#from google.colab import drive
#drive.mount('/content/drive')


# %%
def inspect_instance(instance, mark):
    print("instance {}".format(mark), type(instance))
    print(instance.__class__.__name__)
    for kls in inspect.getmro(instance.__class__):
        print(" ", kls)

# %% [markdown]
# ### Loading data

# %%
from pytorch_forecasting.data.examples import get_stallion_data


# %%
# we want to encode special days as one variable and thus need to first reverse one-hot encoding
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

def get_src_data():
    data = get_stallion_data()
    # add time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
    data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

    data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    return data

df_data = get_src_data()
print(type(df_data))


# %%
df_data.describe()


# %%
df_data.head()


# %%
df_data.tail()


# %%
df_data.sample(10, random_state=521)

# %% [markdown]
# ### Training
# %% [markdown]
# #### dataset

# %%
MAX_PREDICTION_LENGTH = 6
MAX_ENCODER_LENGTH = 24

training_cutoff = df_data["time_idx"].max() - MAX_PREDICTION_LENGTH

training_dataset = TimeSeriesDataSet(
    df_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],

    min_encoder_length=MAX_ENCODER_LENGTH // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=MAX_ENCODER_LENGTH,
    min_prediction_length=1,
    max_prediction_length=MAX_PREDICTION_LENGTH,

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
        "avg_volume_by_sku",],
    
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], 
        # coerce_positive=1.0 --ethan
    ),  # use softplus with beta=1.0 and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, 
                                                    df_data, 
                                                    predict=True, 
                                                    stop_randomization=True) 
                                                    # min_prediction_idx=training_cutoff+1)

# %% [markdown]
# #### dataloaders (train/val)

# %%
# create dataloaders for model
BATCH_SIZE = 128  # set this between 32 to 128
train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)
print(type(train_dataloader))
print(type(val_dataloader))


# %%
def inspect_dataloader(dataloader, name):
    count = 0
    examples = 0
    print("inspect_dataloader", name)
    for x0, x1 in iter(dataloader):
        if count == 0:
            print("iterate dataloader will yield two content: x0 and x1")
            print("x0", type(x0), x0.keys())
            print("x1", type(x1), len(x1), type(x1[0]), x1[0].shape, type(x1[1]))
        count += 1
        examples +=  x1[0].shape[0]
    print("total batches:", count)
    print("total examples:", examples)


# %%
inspect_dataloader(train_dataloader, "train_dataloader")


# %%
inspect_dataloader(val_dataloader, "val_dataloader")

# %% [markdown]
# ### baseline

# %%
def exec_baseline():
    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    #actuals = torch.cat([y for x, y in iter(val_dataloader)])
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]) #ethan
    baseline_predictions = Baseline().predict(val_dataloader)
    baseline_ap_mean = (actuals - baseline_predictions).abs().mean().item()
    print("baseline a-p.mean", baseline_ap_mean)
    return 

baseline_ap_mean = exec_baseline()

# %% [markdown]
# #### trainer

# %%
#resume_from_checkpoint = 'some/path/to/my_checkpoint.ckpt'
resume_from_checkpoint = None


# %%
import os
# configure network and trainer
pl.seed_everything(42)
#os.makedirs("./logs", exist_ok=True)

trainer = pl.Trainer(gpus=1,  #ethan    #gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
    min_epochs=2,
    max_epochs=60, #40, #ethan
    #default_root_dir=".",
    #weights_save_path=".",
    #profiler="advanced",
    resume_from_checkpoint=resume_from_checkpoint,
)
print("trainer", type(trainer))


# %%
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# %%
inspect_instance(tft, "tft")

# %% [markdown]
# #### Training

# %%
def find_optimal_lr():
    global tft, train_dataloader, val_dataloader
    # find optimal learning rate
    res = trainer.tuner.lr_find(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )
    fig = res.plot(show=True, suggest=True)
    fig.show()
    print(f"suggested learning rate: {res.suggestion()}")
    return res.suggestion()


# %%
def train_by_trainer():
    global tft, train_dataloader, val_dataloader
    # can be stopped at any point -- did at 24 epochs
    # fit network
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer.checkpoint_callback.best_model_path


# %%
def optimize_hyperparameters():
    # can cancel, but wait after clicking -- take time to cancel
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=30, #200,
        max_epochs=10,#50
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    import pickle
    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    #print(study.best_trial.params)
    print("study.best_trial.params")
    for key in study.best_trial.params.keys():
        print(" ", key, study.best_trial.params[key])
    return study.best_trial.params


# %%
lr_suggested = find_optimal_lr()


# %%
best_model_path = train_by_trainer()


# %%
#best_trial_params = optimize_hyperparameters()

# %% [markdown]
# ### Forecasting
# %% [markdown]
# #### pickup best_model

# %%
# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
import os
#best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
if os.path.isfile(best_model_path):
    print("##Best model found", best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# %%
inspect_instance(best_tft, "best_tft")


# %%
del tft

# %% [markdown]
# #### Predictions
# 
# %% [markdown]
# ##### Predications - Normal

# %%
# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
besttft_ap_mean = (actuals - predictions).abs().mean()
print("best_tft a-p.mean", besttft_ap_mean)
print("baseline_ap_mean {} vs besttft_ap_mean {}".format(baseline_ap_mean, besttft_ap_mean))


# %%
# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)


# %%
for idx in range(0,3):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)


# %%
for idx in range(3,6):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)


# %%
for idx in range(6,9):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)


# %%
for idx in range(9,12):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

# %% [markdown]
# ##### Predications - Worst

# %%
# worst performer
# calcualte metric by which to display
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses


# %%
for idx in range(0,4):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(4,8):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(8, 12):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(12, 16):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)

# %% [markdown]
# ##### Predictions - Best

# %%
indices = mean_losses.argsort(descending=False)  # sort losses


# %%
for idx in range(0,4):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(4,8):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(8,12):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)


# %%
for idx in range(12,16):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True)

# %% [markdown]
# #### Predication vs actual by variables

# %%
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)


# %%
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

# %% [markdown]
# #### Predication on new data (last 24 months)

# %%
#prediction on new data
# select last 24 months from data (MAX_ENCODER_LENGTH is 24)
encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - MAX_ENCODER_LENGTH]

# select last known data point and create decoder data from it by repeating it and incrementing the month
last_data = data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, MAX_PREDICTION_LENGTH + 1)],
    ignore_index=True,
)

# add time index consistent with "data"
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

# adjust additional time feature(s)
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)


# %%
new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw",return_x=True)


# %%
for idx in range(0,2):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False, add_loss_to_title=True)


# %%
for idx in range(2,4):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False, add_loss_to_title=True)


# %%
for idx in range(4,6):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False, add_loss_to_title=True)


# %%
for idx in range(6,8):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False, add_loss_to_title=True)


# %%
for idx in range(8,10):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False, add_loss_to_title=True)

# %% [markdown]
# #### Attention and Variables (importantance)

# %%
# interpret model
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)

# %% [markdown]
# ## Plot Model

# %%
get_ipython().system('pip install torchviz')


# %%
from torchviz import make_dot


# %%
def print_obj_key(obj, mark=None):
  if mark is not None:
      print(mark)
  for key in dict(obj):
    if not key.startswith("_"):
        print(key)

def get_feed_data():
  for x, y in iter(val_dataloader):
      if first:
        return x


# %%
feed_data = get_feed_data()
resp_out = best_tft.__call__(feed_data)
print_obj_key(resp_out)
print(resp_out["prediction"].shape)


# %%
#It may take a while, and output image maybe too large to load
make_dot(resp_out["prediction"]).render("network", format="png")


# %%
import os
if os.path.isfile("network.png"):
    print("the network structure has been generated")
else:
    print("error")

# %% [markdown]
# ## Summary
# %% [markdown]
# In deep learning, the sequence to sequence approach like RNN and LSTM does shows some promise but required a different architecture and finetuning.
# 
# TFT and N-BEATS are the lastest approaches that are the current state of the arts. They outperformed previous approaches and winner in the M4 competition.
# 

