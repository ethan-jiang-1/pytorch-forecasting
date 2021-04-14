import pickle
import pytorch_forecasting
from pytorch_lightning.core.memory import ModelSummary


class StlTuneHp(object):
    @classmethod
    def turn_hyperparameters(cls, train_dataloader, val_dataloader, n_trials=200, max_epochs=50):
        #from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
        from studies.ce_tuning import optimize_hyperparameters
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
            use_learning_rate_finder=False)

        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)
        return study

    @classmethod 
    def get_hyperparameters(cls, study):
        best_trial = study.best_trial
        best_trial_params = best_trial.params
        print("best_trial_params (copy following if we need keep turnning)")
        for key in sorted(best_trial_params.keys()):
            print("hp.{} = {}".format(key, best_trial_params[key]))
        print("")
        return best_trial.params

    @classmethod
    def get_model_ap_mean(cls, model, val_dataloader):
        import torch
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = model.predict(val_dataloader)
        ap_mean = (actuals - predictions).abs().mean()
        return ap_mean.numpy()

    @classmethod
    def get_baseline_ap_mean(cls, val_dataloader):
        import torch
        from pytorch_forecasting import Baseline        
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = Baseline().predict(val_dataloader)
        ap_mean = (actuals - predictions).abs().mean()
        return ap_mean.numpy()


def _train_by_default_hp(hp, training, train_dataloader, val_dataloader):
    from studies.ce_tft import TftExplorer
    from studies.stl_tft import StlTftExec

    trainer = StlTftExec.get_trainer(hp, max_epochs=3)
    tft = StlTftExec.get_tft_model(hp, training)

    sum = ModelSummary(tft)
    print("\ntft.summary")
    print(sum)
    print("\ntft.hparams")
    print(tft.hparams)

    if hasattr(training, "hack_from_dataset_new_kwargs"):
        opt_explore_tft_from_dataset = True
        if opt_explore_tft_from_dataset:
            if hasattr(training, "hack_from_dataset_new_kwargs"):
                new_kwargs = training.hack_from_dataset_new_kwargs
                TftExplorer.explore_tft_from_dataset_new_kwargs(new_kwargs)

        opt_explore_tft_inputs = True
        if opt_explore_tft_inputs:
            TftExplorer.explore_tft_inputs(training, train_dataloader)

    opt_find_init_lr = True
    if opt_find_init_lr:
        trainer_lr = StlTftExec.get_trainer(hp, max_epochs=1)
        StlTftExec.find_init_lr(trainer_lr, tft, train_dataloader, val_dataloader)

    opt_exec_train_and_pred = True
    if opt_exec_train_and_pred:
        StlTftExec.train(trainer, tft, train_dataloader, val_dataloader)

    return tft


def main():

    from studies.stl_datasrc import StlDataSrc
    from studies.stl_dataloader import StlDataLoader
    from studies.ce_tft import TftExplorer
    from studies.ce_hyperparameters import HyperParameters
    from studies.stl_tft import StlTftExec

    hp = HyperParameters()
    print(hp)

    print(pytorch_forecasting.__version__)

    data = StlDataSrc.get_df_data()
    print(data)

    training = StlDataLoader.get_training_dataset(hp, data)
    validation = StlDataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = StlDataLoader.get_dataloaders(hp, training, validation)

    tft = None
    opt_train_opt = False
    if opt_train_opt:
        tft = _train_by_default_hp(hp, training, train_dataloader, val_dataloader)
        model_ap_mean = StlTuneHp.get_model_ap_mean(tft, val_dataloader)
        baseline_ap_mean = StlTuneHp.get_baseline_ap_mean(val_dataloader)
        print("baseline_ap_mean", baseline_ap_mean)
        print("model_ap_mean", model_ap_mean)

    opt_optimize_hyperparameters = True
    if opt_optimize_hyperparameters:
        study = StlTuneHp.turn_hyperparameters(train_dataloader, val_dataloader, n_trials=2, max_epochs=2)
        print(study)

        best_params = StlTuneHp.get_hyperparameters(study)
        print(best_params)


if __name__ == '__main__':
    main()

