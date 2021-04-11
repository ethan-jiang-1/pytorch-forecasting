from studies.ce_dataloader import DataLoaderExplorer

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
        from studies.ce_dataloader import DataLoaderExplorer
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

            DataLoaderExplorer.explore_dataloader(dataloader, "train")


class TftProfile(object):
    @classmethod
    def profile_training(cls, trainer, tft, train_dataloader, val_dataloader):
        from pytorch_forecasting.utils import profile
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
