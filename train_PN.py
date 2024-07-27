def main(args):
    for training_it in range(args["num_iterations"]):
        
        if training_it > 0:
            args["second_stage"] = True

        import os
        import warnings

        # Suppress warnings
        warnings.filterwarnings("ignore")
        # or '2' to filter out INFO messages too
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import keras
        import tensorflow as tf
        from data_utils import IntermediateEvaluations, create_dataset_from_generator, create_dataset_from_serialized_generator, sequence_dataset_creator, SequenceDataLoader
        from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
        import matplotlib.pyplot as plt

        keras.utils.set_random_seed(args['seed']) # need keras 3 i think

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
        # print all args to file
        with open(os.path.join(RESULTS_SAVE_DIR, "job_args.txt"), "w+") as f:
            for key, value in args.items():
                f.write(f"{key}: {value}\n")

        save_model = True  # if weights will be saved
        plot_intermediate = True  # if the intermediate model predictions will be plotted
        tensorboard = False  # if the Tensorboard callback will be used

        # Training parameters
        if args["model_choice"] == "baseline":
            if args["decompose_images"] or args["object_representations"]: 
                print("Using Baseline PredNet")
                print("Baseline PredNet does not use decomposed images or object representations, disabling both.")
                args["decompose_images"] = False
                args["object_representations"] = False
        if args["decompose_images"]:
            args["output_channels"][0] = 12 # We constrain to the SSM dataset and four object images
        if args["pretrain_classifier"]:
            args["output_channels"] = args["output_channels"][:1] # Just need the bottom layer
            # args["nt"] = 1
            # args["batch_size"] = args["batch_size"] * 16
            # args["sequences_per_epoch_train"] = args["sequences_per_epoch_train"] // 16
            plot_intermediate = False
        nt = args["nt"]  # number of time steps
        nb_epoch = args["nb_epoch"]  # 150
        batch_size = args["batch_size"]  # 4
        # the following two will override the defaults of (dataset size / batch size)
        sequences_per_epoch_train = args["sequences_per_epoch_train"]  # 500
        sequences_per_epoch_val = args["sequences_per_epoch_val"]  # 500
        assert sequences_per_epoch_train is None or type(sequences_per_epoch_train) == int
        assert sequences_per_epoch_val is None or type(sequences_per_epoch_val) == int
        output_channels = args["output_channels"]


        """Create datasets"""
        print("Creating datasets...")
        # Define image shape
        assert args["dataset"] == "SSM", "this branch is focused on the SSM dataset"
        original_im_shape = (args["SSM_im_shape"][0], args["SSM_im_shape"][1], args["output_channels"][0])
        downscale_factor = args["downscale_factor"]
        im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, args["output_channels"][0]) if args["resize_images"] else original_im_shape
        
        train_dataset, train_size = SequenceDataLoader(args, DATA_DIR + "multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_train", nt, batch_size, im_shape[0], im_shape[1], im_shape[2]).create_tf_dataset()
        val_dataset, val_size = SequenceDataLoader(args, DATA_DIR + "multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_val", nt, batch_size, im_shape[0], im_shape[1], im_shape[2]).create_tf_dataset()
        # test_dataset, test_size = SequenceDataLoader(args, DATA_DIR + "multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_test", nt, batch_size, im_shape[0], im_shape[1], im_shape[2]).create_tf_dataset()

        print(f"Working on dataset: {args['dataset']} - {args['data_subset']} {'1st Stage' if not args['second_stage'] else '2nd Stage'}")
        print(f"Train size: {train_size}")
        print(f"Validation size: {val_size}")
        # print(f"Test size: {test_size}")
        print("All datasets created successfully")


        """Model Setup"""
        # PICK MODEL
        if args["model_choice"] == "baseline":
            from PN_models.PN_Baseline import PredNet
        elif args["model_choice"] == "object_centric":
            assert args["decompose_images"], "Object-Centric PredNet requires images to be decomposed"
            from PN_models.PN_ObjectCentric import PredNet
            if args["object_representations"]:
                print("Using the Object-Centric PredNet; Decomposing & classifying inputs, and maintaining & applying object representations.")
            else:
                print("Using the Object-Centric PredNet; Decomposing inputs.")
        else:
            raise ValueError("Invalid model choice")
        
        ###################################
        ##########DEBUG MODE###############
        ###################################
        if args["debug_model"]:
            print("Debugging model...")
            tf.config.run_functions_eagerly(True)
            PN = PredNet(args, im_height=im_shape[0], im_width=im_shape[1])
            # Ensure the model is built by calling it on a sample input or compiling it
            PN.compile(optimizer='adam', loss='mean_squared_error')  # Example optimizer and loss

            # Create a TensorFlow function for optimized execution
            # @tf.function
            def debug_step(x, y):
                with tf.GradientTape() as tape:
                    predictions = PN(x, training=True)  # Get model predictions
                    loss = tf.reduce_mean(tf.square(predictions - y))  # Calculate loss (example)
                gradients = tape.gradient(loss, PN.trainable_variables)
                # Debugging outputs
                tf.print("Loss:", loss)
                # for grad in gradients:
                #     tf.print("Gradient norm:", tf.norm(grad))

            # Iterate over the dataset
            for x, y in train_dataset:
                debug_step(x, y)
        ###################################
        ##########END DEBUG MODE###########
        ###################################
        
        else:
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
            weights_file_name = "baseline_weights.hdf5"
        elif args["model_choice"] == "object_centric" and not args["object_representations"]:
            weights_file_name = "objectCentric_weights.hdf5"
        elif args["model_choice"] == "object_centric" and args["object_representations"]:
            weights_file_name = "objectCentric_withObjectRepresentations_weights.hdf5"
        weights_file = os.path.join(WEIGHTS_DIR, weights_file_name)
        # Define where weights will be saved with results
        results_weights_file = os.path.join(RESULTS_SAVE_DIR, "tensorflow_weights/" + weights_file_name)
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
        else: print("No weights found - starting training from scratch") if not args["restart_training"] else None


        """Training Setup"""
        print(f"Training iteration {training_it+1} of {args['num_iterations']}")

        def lr_schedule(epoch):
            if training_it == 0:
                if epoch == 0:
                    return args["learning_rates"][0]
                elif epoch < 2 * args["nb_epoch"] // 3:
                    return args["learning_rates"][1]
                else:
                    return args["learning_rates"][2]
            elif training_it == 1:
                if epoch < 2 * args["nb_epoch"] // 3:
                    return args["learning_rates"][2]
                else:
                    return args["learning_rates"][3]
            else:
                return args["learning_rates"][3]

        callbacks = [LearningRateScheduler(lr_schedule)]
        if save_model:
            if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR, exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
            callbacks.append(ModelCheckpoint(filepath=results_weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
        if plot_intermediate:
            callbacks.append(IntermediateEvaluations(data_dirs, val_dataset, val_size, batch_size=batch_size, nt=nt, output_channels=output_channels, dataset=args["dataset"], model_choice=args["model_choice"], iteration=training_it+1))
        if tensorboard:
            callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

        history = PN.fit(train_dataset, steps_per_epoch=sequences_per_epoch_train, epochs=nb_epoch, callbacks=callbacks, 
                            validation_data=val_dataset, validation_steps=sequences_per_epoch_val)

if __name__ == "__main__":
    import argparse
    from config import update_settings, get_settings
    import numpy as np
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="PN")  # Training parameters

    # Tuning args
    parser.add_argument("--nt", type=int, default=10, help="sequence length")
    parser.add_argument("--sequences_per_epoch_train", type=int, default=1000, help="number of sequences per epoch for training, otherwise default to dataset size / batch size if None")
    parser.add_argument("--sequences_per_epoch_val", type=int, default=10, help="number of sequences per epoch for validation, otherwise default to validation size / batch size if None")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--nb_epoch", type=int, default=5, help="number of epochs")
    parser.add_argument("--second_stage", type=bool, default=False, help="utilize 2nd stage training data even for first iteration through dataset")

    # Model args
    parser.add_argument("--output_channels", nargs="+", type=int, default=[3, 48, 96, 192], help="output channels. Decompose turns bottom 3 channels to 12")
    parser.add_argument("--downscale_factor", type=int, default=4, help="downscale factor for images prior to training")
    parser.add_argument("--resize_images", type=bool, default=False, help="whether or not to downscale images prior to training")
    parser.add_argument("--decompose_images", type=bool, default=True, help="whether or not to decompose images for training")
    parser.add_argument("--object_representations", type=bool, default=False, help="whether or not to use object representations as input to Rep unit")
    parser.add_argument("--training_split", type=float, default=1, help="proportion of data for training (only for monkaa)")

    # Training args
    parser.add_argument("--seed", type=int, default=np.random.randint(0,1000), help="random seed")
    parser.add_argument("--results_subdir", type=str, default=f"{str(datetime.now())}", help="Specify results directory")
    parser.add_argument("--restart_training", type=bool, default=True, help="whether or not to delete weights and restart")
    parser.add_argument("--reserialize_dataset", type=bool, default=True, help="reserialize dataset")
    parser.add_argument("--output_mode", type=str, default="Error", help="Error, Predictions, or Error_Images_and_Prediction. Only trains on Error.")
    parser.add_argument("--learning_rates", nargs="+", type=int, default=[1e-2, 1e-3, 5e-4, 1e-4], help="output channels")
    parser.add_argument("--pretrain_classifier", type=bool, default=False, help="this will zero out the prediction errors, and focus on the classification diversity loss")
    parser.add_argument("--debug_model", type=bool, default=False, help="this will bypass model.fit and instead feed data through the model to debug the model")

    # Structure args
    parser.add_argument("--model_choice", type=str, default="baseline", help="Choose which model. Options: 'baseline' or 'object_centric'")
    parser.add_argument("--system", type=str, default="laptop", help="laptop or delftblue")
    parser.add_argument("--dataset", type=str, default="SSM", help="SSM - Simple Shape Motion dataset")
    parser.add_argument("--data_subset", type=str, default="multiShape", help="provide descriptive name for results and weights")
    parser.add_argument("--dataset_size", type=int, default=100000, help="total number of images in data dir")
    parser.add_argument("--num_iterations", type=int, default=4, help="number of iterations through the dataset")
    parser.add_argument("--SSM_im_shape", nargs="+", type=int, default=[64, 64], help="output channels")
    """
    Avaialble dataset/data_subset arg combinations:
    - SSM / *: Specify within dataset-creation block which datasets to use. arg["data_subset"] can provide descriptive name for results and weights
    """
    args = parser.parse_args().__dict__

    assert args["training_split"] * args["dataset_size"] >= args["nb_epoch"] * args["sequences_per_epoch_train"] * args["batch_size"], "Ensure that the data length in each training iteration is not exceeded"

    update_settings(args["system"], args["dataset"], args["data_subset"], args["results_subdir"])
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = get_settings()["dirs"]
    data_dirs = [DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR]

    main(args)
