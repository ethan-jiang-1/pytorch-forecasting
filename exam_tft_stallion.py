from studies.stl_datasrc import StlDataSrc
from studies.stl_dataloader import StlDataLoader
from studies.stl_tft import StlTftExec
from studies.ce_hyperparameters import HyperParameters

def main():

    hp = HyperParameters()

    data = StlDataSrc.get_df_data()
    print(data)

    training = StlDataLoader.get_training_dataset(hp, data)
    validation = StlDataLoader.get_validation_dataset(training, data)

    train_dataloader, val_dataloader = StlDataLoader.get_dataloaders(hp, training, validation)

    trainer = StlTftExec.get_trainer(hp, max_epochs=3)
    tft = StlTftExec.get_tft_model(hp, training)

    StlTftExec.find_init_lr(trainer, tft, train_dataloader, val_dataloader)
    StlTftExec.train(trainer, tft, train_dataloader, val_dataloader)
    study = StlTftExec.turn_hyperparameters(train_dataloader, val_dataloader, n_trials=2, max_epochs=2)
    print(study)

    preds, index = StlTftExec.predict(tft, val_dataloader)
    print(preds)
    print(index)


if __name__ == '__main__':
    main()
