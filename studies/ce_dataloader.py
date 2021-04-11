
from torch.utils.tensorboard import SummaryWriter

class DataLoaderExplorer(object):
    @classmethod
    def explore_dataset(cls, dataset, name):
        print("")
        print("##DataExplorer.explore_dataset", name)
        parameters = dataset.get_parameters()
        print("dataset {} parameters".format(name))
        for key in parameters.keys():
            print("  ", key, "=", parameters[key], " / ", type(parameters[key]))
        print("")

        print("*dataset.scalars:")
        for sl in dataset.scalers:
            print("  ", sl, dataset.scalers[sl])
        print("*dataset.categorical_encoders")
        for ce in dataset.categorical_encoders:
            print("  ", ce, dataset.categorical_encoders[ce])

        print("")
        print("*(V)flat_categoricals", len(dataset.flat_categoricals))
        for fc in dataset.flat_categoricals:
            print("  ", fc)
        
        print("")
        print("*(V)reals:", len(dataset.reals))
        for rl in dataset.reals:
            print("  ", rl)

    @classmethod
    def explore_dataloader(cls, dataloader, name):
        writer = SummaryWriter()

        count = 0
        examples = 0
        print("")
        print("##DataExplorer.explore_dataloader", name)
    
        for x0, x1 in iter(dataloader):
            if count == 0:
                print("encoder_cat.shape", x0["encoder_cat"].shape)
                print("encoder_cont.shape", x0["encoder_cont"].shape)
                print("encoder_target.shape", x0["encoder_target"].shape)
                print("encoder_lengths.shape", x0["encoder_lengths"].shape)
                print("")
                print("decoder_cat.shape", x0["decoder_cat"].shape)
                print("decoder_cont.shape", x0["decoder_cont"].shape)
                print("decoder_target.shape", x0["decoder_target"].shape)
                print("decoder_lengths.shape", x0["decoder_lengths"].shape)
                print("")
                print("decoder_time_idx.shape", x0["decoder_time_idx"].shape)
                print("groups.shape", x0["groups"].shape)
                print("target_scale.shape", x0["target_scale"].shape)

                print("x1_0.shape", x1[0].shape)
                print("x1_1", None)
            elif count == 1:
                print("iterating dataloader...")

            if count % 10 == 0:
                import sys
                sys.stdout.write("...{}...\r".format(count))
                sys.stdout.flush()
            count += 1
            examples += x1[0].shape[0]

        print("")
        print("Summarize the dataloader output nums:", name)
        print("total batches:", count)
        print("total examples:", examples)

        writer.close()
