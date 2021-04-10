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

#from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
#from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data
#from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
#from pytorch_forecasting.utils import profile

#import warnings
#warnings.simplefilter("error", category=SettingWithCopyWarning)

from torch.utils.tensorboard import SummaryWriter

def test_sw():
    import numpy as np

    writer = SummaryWriter()

    for n_iter in range(200,300):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    r = 5
    for i in range(200,300):
        writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                       'xcosx':i*np.cos(i/r),
                                       'tanx': np.tan(i/r)}, i)

    for i in range(10,20):
        writer.add_hparams({'lr': 0.1*i, 'bsize': i},
                        {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

    writer.add_text('lstm', 'This is an lstm', 1)
    writer.add_text('rnn', 'This is an rnn', 11)

    dummy_data = []
    for idx, value in enumerate(range(50)):
        dummy_data += [idx + 0.001] * value

    bins = list(range(50+2))
    bins = np.array(bins)
    values = np.array(dummy_data).astype(float).reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    sum_sq = values.dot(values)
    writer.add_histogram_raw(
        tag='histogram_with_raw_data',
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limits=limits[1:].tolist(),
        bucket_counts=counts.tolist(),
        global_step=0)

    writer.close()


def main():
    test_sw()


if __name__ == '__main__':
    main()
