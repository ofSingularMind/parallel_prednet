def main(args):
    for training_it in range(args["num_dataset_chunks"]):
        # uncomment to resume training at a specific iteration, because it crashes
        if (training_it+1) < args["train_restart_iteration"]: continue

        import os
        import warnings

        # Suppress warnings
        warnings.filterwarnings("ignore")
        # or '2' to filter out INFO messages too
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import keras
        import tensorflow as tf
        from data_utils import IntermediateEvaluations, create_dataset_from_generator, create_dataset_from_serialized_generator 
        from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

        keras.utils.set_random_seed(args['seed']) # need keras 3 i think

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
        # print all args to file
        with open(os.path.join(RESULTS_SAVE_DIR, "job_args.txt"), "w+") as f:
            for key, value in args.items():
                f.write(f"{key}: {value}\n")

        save_model = True  # if weights will be saved
        plot_intermediate = True  # if the intermediate model predictions will be plotted
        tensorboard = True  # if the Tensorboard callback will be used

        # Training parameters
        nt = args["nt"]  # number of time steps
        nb_epoch = args["nb_epoch"]  # 150
        batch_size = args["batch_size"]  # 4
        # the following two will override the defaults of (dataset size / batch size)
        sequences_per_epoch_train = args["sequences_per_epoch_train"]  # 500
        sequences_per_epoch_val = args["sequences_per_epoch_val"]  # 500
        assert sequences_per_epoch_train is None or type(sequences_per_epoch_train) == int
        assert sequences_per_epoch_val is None or type(sequences_per_epoch_val) == int
        if args["decompose_images"]:
            args["output_channels"][0] = 12 # We constrain to the SSM dataset and four object images
        output_channels = args["output_channels"]


        """Model Setup"""
        # PICK MODEL
        if args["model_choice"] == "baseline":
            assert not args["decompose_images"], "Baseline PredNet does not use decomposed images"
            # Predict next frame along RGB channels only
            from PN_models.PN_Baseline import PredNet
            print("Using Baseline PredNet")
        elif args["model_choice"] == "object_centric":
            assert args["decompose_images"], "Object-Centric PredNet requires images to be decomposed"
            from PN_models.PN_ObjectCentric import PredNet
            if args["object_representations"]:
                print("Using the Object-Centric PredNet; Decomposing & classifying inputs, and maintaining & applying object representations.")
            else:
                print("Using the Object-Centric PredNet; Decomposing inputs.")
        else:
            raise ValueError("Invalid model choice")

        # Define image shape
        assert args["dataset"] == "SSM", "this branch is focused on the SSM dataset"
        original_im_shape = (args["SSM_im_shape"][0], args["SSM_im_shape"][1], args["output_channels"][0])
        downscale_factor = args["downscale_factor"]
        im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, args["output_channels"][0]) if args["resize_images"] else original_im_shape
        
        # Create PredNet with animation specific input shapes
        inputs = keras.Input(shape=(nt, im_shape[0], im_shape[1], im_shape[2]), batch_size=batch_size)
        PN = PredNet(args, im_height=im_shape[0], im_width=im_shape[1])
        outputs = PN(inputs)
        PN = keras.Model(inputs=inputs, outputs=outputs)
        
        # Finalize model
        resos = PN.layers[-1].resolutions
        PN.compile(optimizer="adam", loss="mean_squared_error")
        print("PredNet compiled...")
        PN.build(input_shape=(batch_size, nt) + im_shape)
        print(PN.summary())
        num_layers = len(output_channels)  # number of layers in the architecture
        print(f"{num_layers} PredNet layers with resolutions:")
        for i in reversed(range(num_layers)):
            print(f"Layer {i+1}:  {resos[i][0]} x {resos[i][1]} x {output_channels[i]}")


        """Weights Setup"""
        # Define where weights will be loaded/saved
        if args["model_choice"] == "baseline":
            file_name = "baseline_weights.hdf5"
        elif args["model_choice"] == "object_centric" and not args["object_representations"]:
            file_name = "objectCentric_weights.hdf5"
        elif args["model_choice"] == "object_centric" and args["object_representations"]:
            file_name = "objectCentric_withObjectRepresentations_weights.hdf5"
        weights_file = os.path.join(WEIGHTS_DIR, file_name)
        # Define where weights will be saved with results
        results_weights_file = os.path.join(RESULTS_SAVE_DIR, "tensorflow_weights/"+file_name)
        # Remove weights file if restarting training. Previous weights can still be found with the results
        if args["restart_training"] and training_it == 0:
            if os.path.exists(weights_file):
                os.remove(weights_file)
            print("Restarting training from scratch")
        # load previously saved weights
        if os.path.exists(weights_file):
            try: 
                PN.load_weights(weights_file)
                print("Weights loaded successfully - continuing training from last epoch")
            except: 
                os.remove(weights_file) # model architecture has changed, so weights cannot be loaded
                print("Weights don't fit - restarting training from scratch")
        elif args["restart_training"]:
            print("Restarting training from scratch")
        else: print("No weights found - starting training from scratch")
 

        """Create datasets"""
        # dataset_names = ["general_shape_strafing", "general_shape_strafing"]
        # data_subset_names = ["general_cross_R","general_ellipse_D",]
        dataset_names = ["multi_gen_shape_strafing"]
        data_subset_names = ["multi_gen_shape_1st_stage" if not args["second_stage"] else "multi_gen_shape_2nd_stage"]
        # dataset_names = ["class_cond_shape_strafing"]
        # data_subset_names = ["class_cond_shape_1st_stage" if not args["second_stage"] else "class_cond_shape_2nd_stage"]
        # dataset_names = ["world_cond_shape_strafing"]
        # data_subset_names = ["world_cond_shape_1st_stage" if not args["second_stage"] else "world_cond_shape_2nd_stage"]

        # print dataset names to job details file
        with open(os.path.join(RESULTS_SAVE_DIR, "job_args.txt"), "a+") as f:
            f.write(f"Dataset names: {dataset_names}\n")

        # Training data
        list_png_paths = []
        for ds_name, dss_name in zip(dataset_names, data_subset_names):
            assert os.path.exists(DATA_DIR + f"{ds_name}/frames/{dss_name}/" + "/001.png"), "Dataset not found"
            list_png_paths.append([DATA_DIR + f"{ds_name}/frames/{dss_name}/"]) # 3 channels (RGB)

        train_split = args["training_split"]
        val_split = (1 - train_split) / 2

        length = 0
        full_train_dataset, full_val_dataset, full_test_dataset = None, None, None
        for png_paths, dataset_name in zip(list_png_paths, dataset_names):
            #  Create and split dataset
            datasets, ds_len = create_dataset_from_serialized_generator(data_dirs, png_paths, output_mode="Error", dataset_name=dataset_name, im_height=im_shape[0], im_width=im_shape[1],
                                                                        output_channels=im_shape[2], batch_size=batch_size, nt=nt, train_split=train_split, reserialize=args["reserialize_dataset"], 
                                                                        shuffle=True, resize=args["resize_images"], single_channel=False, iteration=training_it, decompose=args["decompose_images"])
            train_dataset, val_dataset, test_dataset = datasets
            full_train_dataset = train_dataset if full_train_dataset is None else full_train_dataset.concatenate(train_dataset)
            full_val_dataset = val_dataset if full_val_dataset is None else full_val_dataset.concatenate(val_dataset)
            full_test_dataset = test_dataset if full_test_dataset is None else full_test_dataset.concatenate(test_dataset)
            length += ds_len


        train_size = int(train_split * length)
        val_size = int(val_split * length)
        test_size = length-train_size-val_size

        full_train_dataset = full_train_dataset.shuffle(train_size, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat() # .shuffle(train_size)
        full_val_dataset = full_val_dataset.shuffle(val_size, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat() # .shuffle(val_size)
        full_test_dataset = full_test_dataset.shuffle(test_size, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat() # .shuffle(test_size)

        train_dataset, val_dataset, test_dataset = full_train_dataset, full_val_dataset, full_test_dataset
        
        print(f"Working on dataset: {args['dataset']} - {args['data_subset']} {'1st Stage' if not args['second_stage'] else '2nd Stage'}")
        print(f"Train size: {train_size}")
        print(f"Validation size: {val_size}")
        print(f"Test size: {test_size}")
        print("All datasets created successfully")


        """Training Setup"""
        print(f"Training iteration {training_it+1} of {args['num_dataset_chunks']}")
        
        # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
        def lr_schedule(epoch): 
            if training_it == 0:
                return args["learning_rates"][0]
            elif 0 < training_it and training_it < args["num_dataset_chunks"] // 2:
                return args["learning_rates"][1] 
            # elif 50 < epoch <= 100:
            #     return args["learning_rates"][2]
            else:
                return args["learning_rates"][3]

        callbacks = [LearningRateScheduler(lr_schedule)]
        if save_model:
            if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR, exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
            callbacks.append(ModelCheckpoint(filepath=results_weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
        if plot_intermediate:
                callbacks.append(IntermediateEvaluations(data_dirs, test_dataset, test_size, batch_size=batch_size, nt=nt, output_channels=output_channels, dataset=args["dataset"], model_choice=args["model_choice"], iteration=training_it+1))
        if tensorboard:
            callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

        history = PN.fit(train_dataset, steps_per_epoch=train_size // batch_size if sequences_per_epoch_train is None else sequences_per_epoch_train,
                        epochs=nb_epoch, callbacks=callbacks, validation_data=val_dataset, validation_steps=val_size // batch_size if sequences_per_epoch_val is None else sequences_per_epoch_val)

if __name__ == "__main__":
    import argparse
    from config import update_settings, get_settings
    import numpy as np
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="PN")  # Training parameters

    # Tuning args
    parser.add_argument("--nt", type=int, default=10, help="sequence length")
    parser.add_argument("--sequences_per_epoch_train", type=int, default=800, help="number of sequences per epoch for training, otherwise default to dataset size / batch size if None")
    parser.add_argument("--sequences_per_epoch_val", type=int, default=10, help="number of sequences per epoch for validation, otherwise default to validation size / batch size if None")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--nb_epoch", type=int, default=1, help="number of epochs")
    parser.add_argument("--second_stage", type=bool, default=True, help="utilize 2nd stage training data")
    """
    unser bs 20 x 25 steps = 42 sec -> 11.9 sequences/sec
    unser bs 30 x 10 steps = 24 sec -> 12.5 sequences/sec ***
    ser bs 4 x 25 steps = 13 sec -> 6.76 sequences/sec
    """
    parser.add_argument("--output_channels", nargs="+", type=int, default=[3, 48, 96, 192], help="output channels. Decompose turns bottom 3 channels to 12")
    parser.add_argument("--downscale_factor", type=int, default=4, help="downscale factor for images prior to training")
    parser.add_argument("--resize_images", type=bool, default=False, help="whether or not to downscale images prior to training")
    parser.add_argument("--decompose_images", type=bool, default=True, help="whether or not to decompose images for training")
    parser.add_argument("--object_representations", type=bool, default=True, help="whether or not to use object representations as input to Rep unit")
    parser.add_argument("--training_split", type=float, default=0.80, help="proportion of data for training (only for monkaa)")

    # Training args
    parser.add_argument("--seed", type=int, default=np.random.randint(0,1000), help="random seed")
    parser.add_argument("--results_subdir", type=str, default=f"{str(datetime.now())}", help="Specify results directory")
    parser.add_argument("--restart_training", type=bool, default=False, help="whether or not to delete weights and restart")
    parser.add_argument("--reserialize_dataset", type=bool, default=True, help="reserialize dataset")
    parser.add_argument("--output_mode", type=str, default="Error", help="Error, Predictions, or Error_Images_and_Prediction. Only trains on Error.")
    parser.add_argument("--train_restart_iteration", type=int, default=6, help="when training crashes, can restart from last iteration. 0 means to start from the beginning")
    parser.add_argument("--learning_rates", nargs="+", type=int, default=[1e-3, 5e-4, 99, 1e-4], help="output channels")

    # Structure args
    parser.add_argument("--model_choice", type=str, default="object_centric", help="Choose which model. Options: 'baseline' or 'object_centric'")
    parser.add_argument("--system", type=str, default="laptop", help="laptop or delftblue")
    parser.add_argument("--dataset", type=str, default="SSM", help="SSM - Simple Shape Motion dataset")
    parser.add_argument("--data_subset", type=str, default="multiShape", help="family_x2 only for laptop, any others (ex. treeflight_x2) for delftblue")
    parser.add_argument("--num_dataset_chunks", type=int, default=20, help="number of dataset chunks to iterate through (full DS / 2000)")
    parser.add_argument("--SSM_im_shape", nargs="+", type=int, default=[64, 64], help="output channels")
    """
    Avaialble dataset/data_subset arg combinations:
    - SSM / *: Specify within dataset-creation block which datasets to use. arg["data_subset"] can provide descriptive name for results and weights
    """
    args = parser.parse_args().__dict__

    update_settings(args["system"], args["dataset"], args["data_subset"], args["results_subdir"])
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = get_settings()["dirs"]
    data_dirs = [DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR]

    main(args)
