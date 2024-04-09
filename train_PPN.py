def main(args):
    import os
    import warnings

    # Suppress warnings
    warnings.filterwarnings("ignore")
    # or '2' to filter out INFO messages too
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf
    import shutil
    import keras
    from data_utils import SequenceGenerator, IntermediateEvaluations, create_dataset_from_serialized_generator, config_gpus 
    from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

    # PICK MODEL
    if args["model_choice"] == "baseline":
        # Predict next frame along RGB channels only
        if not args['pan_hierarchical']:
            from PPN_models.PPN_Baseline import ParaPredNet
        else:
            from PPN_models.PPN_Baseline import ParaPredNet
            print("Using Pan-Hierarchical Representation")
    elif args["model_choice"] == "cl_delta":
        # Predict next frame and change from current frame
        from PPN_models.PPN_CompLearning_Delta_Predictions import ParaPredNet
    elif args["model_choice"] == "cl_recon":
        # Predict current and next frame
        from PPN_models.PPN_CompLearning_Recon_Predictions import ParaPredNet
    elif args["model_choice"] == "multi_channel":
        # Predict next frame along Disparity, Material Index, Object Index, 
        # Optical Flow, Motion Boundaries, and RGB channels all stacked together
        assert args["dataset"] == "monkaa" or args["dataset"] == "driving", "Multi-channel model only works with Monkaa or Driving dataset"
        from PPN_models.PPN_Multi_Channel import ParaPredNet
        bottom_layer_output_channels = 7 # 1 Disparity, 3 Optical Flow, 3 RGB
        args["output_channels"][0] = bottom_layer_output_channels
    else:
        raise ValueError("Invalid model choice")


    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) backend random seed
    # 3) `python` random seed
    keras.utils.set_random_seed(args['seed']) # need keras 3 i think

    # use mixed precision for faster runtimes and lower memory usage
    # keras.mixed_precision.set_global_policy("mixed_float16")
    # config_gpus()

    # if results directory already exists, then delete it
    if os.path.exists(RESULTS_SAVE_DIR):
        shutil.rmtree(RESULTS_SAVE_DIR)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    # print all args to file
    with open(os.path.join(RESULTS_SAVE_DIR, "job_args.txt"), "w+") as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")

    save_model = True  # if weights will be saved
    plot_intermediate = True  # if the intermediate model predictions will be plotted
    tensorboard = True  # if the Tensorboard callback will be used

    if (args["dataset"], args["data_subset"]) in [
        ("rolling_square", "single_rolling_square"),
        ("rolling_circle", "single_rolling_circle")
    ]:
        # where weights will be loaded/saved
        weights_file = os.path.join(WEIGHTS_DIR, f"para_prednet_"+args["data_subset"]+"_weights.hdf5")
        # where weights will be saved with results
        results_weights_file = os.path.join(RESULTS_SAVE_DIR, f"tensorflow_weights/para_prednet_"+args["data_subset"]+"_weights.hdf5")
    
    else:
        # where weights will be loaded/saved
        weights_file = os.path.join(WEIGHTS_DIR, f"para_prednet_"+args["dataset"]+"_weights.hdf5")
        # where weights will be saved with results
        results_weights_file = os.path.join(RESULTS_SAVE_DIR, f"tensorflow_weights/para_prednet_"+args["dataset"]+"_weights.hdf5")
    
    if args["restart_training"]:
        if os.path.exists(weights_file):
            os.remove(weights_file)

    # Training parameters
    nt = args["nt"]  # number of time steps
    nb_epoch = args["nb_epoch"]  # 150
    batch_size = args["batch_size"]  # 4
    # the following two will override the defaults of (dataset size / batch size)
    sequences_per_epoch_train = args["sequences_per_epoch_train"]  # 500
    sequences_per_epoch_val = args["sequences_per_epoch_val"]  # 500
    assert sequences_per_epoch_train is None or type(sequences_per_epoch_train) == int
    assert sequences_per_epoch_val is None or type(sequences_per_epoch_val) == int
    num_P_CNN = args["num_P_CNN"]
    num_R_CLSTM = args["num_R_CLSTM"]
    output_channels = args["output_channels"]

    # Define image shape
    if args["dataset"] == "kitti":
        original_im_shape = (128, 160, 3)
        im_shape = original_im_shape
    elif args["dataset"] == "monkaa" or args["dataset"] == "driving":
        original_im_shape = (540, 960, 3)
        downscale_factor = args["downscale_factor"]
        im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, 3)
    elif args["dataset"] in ["rolling_square", "rolling_circle"]:
        original_im_shape = (50, 100, 3)
        downscale_factor = args["downscale_factor"]
        im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, 3) if args["resize_images"] else original_im_shape

    # Create datasets
    if args["dataset"] == "kitti":
        # Data files
        train_file = os.path.join(DATA_DIR, "X_train.hkl")
        train_sources = os.path.join(DATA_DIR, "sources_train.hkl")
        val_file = os.path.join(DATA_DIR, "X_val.hkl")
        val_sources = os.path.join(DATA_DIR, "sources_val.hkl")
        test_file = os.path.join(DATA_DIR, "X_test.hkl")
        test_sources = os.path.join(DATA_DIR, "sources_test.hkl")

        train_dataset = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
        val_dataset = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=len(val_sources) // batch_size if sequences_per_epoch_val is None else sequences_per_epoch_val, shuffle=False)
        test_dataset = SequenceGenerator(test_file, test_sources, nt, batch_size=batch_size, shuffle=False)
        train_size = train_dataset.N_sequences
        val_size = val_dataset.N_sequences
        test_size = test_dataset.N_sequences
        # print("All generators created successfully")

    elif args["dataset"] == "monkaa":
        # Training data
        assert os.path.exists(DATA_DIR + "disparity/" + args["data_subset"] + "/left/"), "Improper data_subset selected"
        pfm_paths = []
        pfm_paths.append(DATA_DIR + "disparity/" + args["data_subset"] + "/left/") # 1 channel
        pfm_paths.append(DATA_DIR + "material_index/" + args["data_subset"] + "/left/") # 1 channel
        pfm_paths.append(DATA_DIR + "object_index/" + args["data_subset"] + "/left/") # 1 channel
        pfm_paths.append(DATA_DIR + "optical_flow/" + args["data_subset"] + "/into_future/left/") # 3 channels
        pgm_paths = []
        pgm_paths.append(DATA_DIR + "motion_boundaries/" + args["data_subset"] + "/into_future/left/") # 1 channel
        png_paths = []
        png_paths.append(DATA_DIR + "frames_cleanpass/" + args["data_subset"] + "/left") # 3 channels (RGB)
        num_sources = len(pfm_paths) + len(pgm_paths) + len(png_paths)

        train_split = 0.7
        val_split = (1 - train_split) / 2
        #  Create and split dataset
        datasets, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode="Error", im_height=im_shape[0], im_width=im_shape[1],
                                                                    batch_size=batch_size, nt=nt, train_split=train_split, reserialize=args["reserialize_dataset"], shuffle=True, resize=True)
        train_dataset, val_dataset, test_dataset = datasets

        train_size = int(train_split * length)
        val_size = int(val_split * length)
        test_size = int(val_split * length)

    elif args["dataset"] == "driving":
        # Training data
        assert os.path.exists(DATA_DIR + "disparity/15mm_focallength/scene_forwards/slow/left/"), "Dataset not found"
        pfm_paths = []
        pfm_paths.append(DATA_DIR + "disparity/15mm_focallength/scene_forwards/slow/left/") # 1 channel
        pfm_paths.append(DATA_DIR + "optical_flow/15mm_focallength/scene_forwards/slow/into_future/left/") # 3 channels
        pgm_paths = []
        png_paths = []
        png_paths.append(DATA_DIR + "frames_cleanpass/15mm_focallength/scene_forwards/slow/left") # 3 channels (RGB)
        num_sources = len(pfm_paths) + len(pgm_paths) + len(png_paths)

        train_split = args["training_split"]
        val_split = (1 - train_split) / 2
        #  Create and split dataset
        datasets, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode="Error", dataset_name="driving", im_height=im_shape[0], im_width=im_shape[1],
                                                                    batch_size=batch_size, nt=nt, train_split=train_split, reserialize=args["reserialize_dataset"], shuffle=True, resize=args["resize_images"])
        train_dataset, val_dataset, test_dataset = datasets

        train_size = int(train_split * length)
        val_size = int(val_split * length)
        test_size = int(val_split * length)

    elif args["dataset"] in ["rolling_square", "rolling_circle"]:
        # Training data
        assert os.path.exists(DATA_DIR + "/001.png"), "Dataset not found"
        pfm_paths = []
        pgm_paths = []
        png_paths = []
        png_paths.append(DATA_DIR) # 3 channels (RGB)
        num_sources = len(pfm_paths) + len(pgm_paths) + len(png_paths)

        train_split = args["training_split"]
        val_split = (1 - train_split) / 2
        #  Create and split dataset
        datasets, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode="Error", dataset_name=args["data_subset"], im_height=im_shape[0], im_width=im_shape[1],
                                                                    batch_size=batch_size, nt=nt, train_split=train_split, reserialize=args["reserialize_dataset"], shuffle=True, resize=args["resize_images"], single_channel=False)
        train_dataset, val_dataset, test_dataset = datasets

        train_size = int(train_split * length)
        val_size = int(val_split * length)
        test_size = int(val_split * length)

    print(f"Working on dataset: {args['dataset']}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")
    print("All datasets created successfully")

    # Create ParaPredNet
    if args["dataset"] == "kitti":
        # These are Kitti specific input shapes
        inputs = (keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)))
        PPN = ParaPredNet(args, im_height=im_shape[0], im_width=im_shape[1])  # [3, 48, 96, 192]
        outputs = PPN(inputs)
        PPN = keras.Model(inputs=inputs, outputs=outputs)

    elif args["dataset"] == "monkaa":
        # These are Monkaa specific input shapes
        inputs = (keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
        )
        PPN = ParaPredNet(args, im_height=im_shape[0], im_width=im_shape[1])  # [3, 48, 96, 192]
        outputs = PPN(inputs)
        PPN = keras.Model(inputs=inputs, outputs=outputs)

    elif args["dataset"] == "driving":
        # These are driving specific input shapes
        inputs = (keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
            keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
        )
        PPN = ParaPredNet(args, im_height=im_shape[0], im_width=im_shape[1])  # [3, 48, 96, 192]
        outputs = PPN(inputs)
        PPN = keras.Model(inputs=inputs, outputs=outputs)

    elif args["dataset"] in ["rolling_square", "rolling_circle"]:
        # These are animation specific input shapes
        inputs = keras.Input(shape=(nt, im_shape[0], im_shape[1], 3))
        PPN = ParaPredNet(args, im_height=im_shape[0], im_width=im_shape[1])
        outputs = PPN(inputs)
        PPN = keras.Model(inputs=inputs, outputs=outputs)
    
    resos = PPN.layers[-1].resolutions
    PPN.compile(optimizer="adam", loss="mean_squared_error")
    print("ParaPredNet compiled...")
    PPN.build(input_shape=(None, nt) + im_shape)
    print(PPN.summary())
    num_layers = len(output_channels)  # number of layers in the architecture
    print(f"{num_layers} PredNet layers with resolutions:")
    for i in reversed(range(num_layers)):
        print(f"Layer {i+1}:  {resos[i][0]} x {resos[i][1]} x {output_channels[i]}")

    # load previously saved weights
    if os.path.exists(weights_file):
        try: 
            PPN.load_weights(weights_file)
            print("Weights loaded successfully - continuing training from last epoch")
        except: 
            os.remove(weights_file) # model architecture has changed, so weights cannot be loaded
            print("Weights don't fit - restarting training from scratch")
    elif args["restart_training"]:
        print("Restarting training from scratch")
    else: print("No weights found - starting training from scratch")

    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    def lr_schedule(epoch): 
        if epoch <= 50:
            return args["learning_rates"][0] 
        # elif 5 < epoch <= 10:
        #     return args["learning_rates"][1] 
        # elif 10 < epoch <= 20:
        #     return args["learning_rates"][2]
        else:
            return args["learning_rates"][1]

    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR, exist_ok=True)
        callbacks.append(ModelCheckpoint(filepath=weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
        callbacks.append(ModelCheckpoint(filepath=results_weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
    if plot_intermediate:
            callbacks.append(IntermediateEvaluations(test_dataset, test_size, batch_size=batch_size, nt=nt, output_channels=output_channels, dataset=args["dataset"], model_choice=args["model_choice"]))
    if tensorboard:
        callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

    history = PPN.fit(train_dataset, steps_per_epoch=train_size // batch_size if sequences_per_epoch_train is None else sequences_per_epoch_train,
                      epochs=nb_epoch, callbacks=callbacks, validation_data=val_dataset, validation_steps=val_size // batch_size if sequences_per_epoch_val is None else sequences_per_epoch_val)


if __name__ == "__main__":
    import argparse
    from config import update_settings, get_settings
    import numpy as np
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="PPN")  # Training parameters

    # Tuning args
    parser.add_argument("--nt", type=int, default=10, help="sequence length")
    parser.add_argument("--nb_epoch", type=int, default=250, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size")
    parser.add_argument("--output_channels", nargs="+", type=int, default=[3, 6, 12, 24], help="output channels")
    parser.add_argument("--sequences_per_epoch_train", type=int, default=200, help="number of sequences per epoch for training, otherwise default to dataset size / batch size if None")
    parser.add_argument("--sequences_per_epoch_val", type=int, default=None, help="number of sequences per epoch for validation, otherwise default to validation size / batch size if None")
    parser.add_argument("--num_P_CNN", type=int, default=1, help="number of serial Prediction convolutions")
    parser.add_argument("--num_R_CLSTM", type=int, default=1, help="number of hierarchical Representation CLSTMs")
    parser.add_argument("--num_passes", type=int, default=1, help="number of prediction-update cycles per time-step")
    parser.add_argument("--pan_hierarchical", type=bool, default=False, help="utilize Pan-Hierarchical Representation")
    parser.add_argument("--downscale_factor", type=int, default=4, help="downscale factor for images prior to training")
    parser.add_argument("--resize_images", type=bool, default=False, help="whether or not to downscale images prior to training")
    parser.add_argument("--training_split", type=float, default=0.7, help="proportion of data for training (only for monkaa)")

    # Training args
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument("--results_subdir", type=str, default=f"{str(datetime.now())}", help="Specify results directory")
    parser.add_argument("--restart_training", type=bool, default=False, help="whether or not to delete weights and restart")
    parser.add_argument("--learning_rates", nargs="+", type=int, default=[5e-4, 5e-4, 1e-4, 5e-5], help="output channels")

    # Structure args
    parser.add_argument("--model_choice", type=str, default="baseline", help="Choose which model. Options: baseline, cl_delta, cl_recon, multi_channel")
    parser.add_argument("--system", type=str, default="laptop", help="laptop or delftblue")
    parser.add_argument("--dataset", type=str, default="rolling_circle", help="kitti, driving, monkaa, rolling_square, or rolling_circle")
    parser.add_argument("--data_subset", type=str, default="single_rolling_circle", help="family_x2 only for laptop, any others (ex. treeflight_x2) for delftblue")
    parser.add_argument("--reserialize_dataset", type=bool, default=False, help="reserialize dataset")
    parser.add_argument("--output_mode", type=str, default="Error", help="Error, Predictions, or Error_Images_and_Prediction (only trains on Error)")
    
    args = parser.parse_args().__dict__

    update_settings(args["system"], args["dataset"], args["data_subset"], args["results_subdir"])
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = get_settings()["dirs"]
    
    main(args)
