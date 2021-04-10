from studies.ds_stallion import DataSrc, DataLoader
from studies.tft_model import TftExec


def main():

    data = DataSrc.get_df_data()
    print(data)

    training = DataLoader.get_training_dataset(data)
    validation = DataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = DataLoader.get_dataloaders(training, validation)

    trainer = TftExec.get_trainer(max_epochs=3)
    tft = TftExec.get_tft_model(training)

    TftExec.find_init_lr(trainer, tft, train_dataloader, val_dataloader)
    TftExec.train(trainer, tft, train_dataloader, val_dataloader)
    study = TftExec.turn_hyperparameters(train_dataloader, val_dataloader, n_trials=2, max_epochs=2)
    print(study)

    preds, index = TftExec.predict(tft, val_dataloader)
    print(preds)
    print(index)


if __name__ == '__main__':
    main()
