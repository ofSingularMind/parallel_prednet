def main(args):
    for training_it in range(args["num_iterations"]):

        if (training_it+1) < args["restart_iteration"]:
            continue

        elif (training_it+1) > args["restart_iteration"]:
            args["restart_training"] = False
        
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
                print("Baseline PredNet does not use decomposed images or object representations, disabling both.")
                args["decompose_images"] = False
                args["object_representations"] = False
        if args["object_representations"]:
            # args["batch_size"] = 1
            args["include_frame"] = False
        if args["decompose_images"]:
            args["output_channels"][0] = 12 # We constrain to the SSM dataset and four object images
            if args["include_frame"]:
                args["output_channels"][0] += 3 # Add the original frame channels
        if args["pretrain_latent_lpn"]:
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

        if args["debug_model"]:
            print("Debugging model...")
            tf.config.run_functions_eagerly(True)

        """Create datasets"""
        print("Creating datasets...")
        # Define image shape
        assert args["dataset"] == "SSM", "this branch is focused on the SSM dataset"
        original_im_shape = (args["dataset_im_shape"][0], args["dataset_im_shape"][1], args["output_channels"][0])
        downscale_factor = args["downscale_factor"]
        im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, args["output_channels"][0]) if args["resize_images"] else original_im_shape
        
        # Always 2nd stage training for object-centric model because the random backgrounds are applied at decomposition time
        stage = ("2nd_stage" if args["second_stage"] else "1st_stage") if args["model_choice"] == "baseline" else "2nd_stage"
        train_dataset, train_size = SequenceDataLoader(args, DATA_DIR + f"multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_train", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, args["include_frame"]).create_tf_dataset()
        val_dataset, val_size = SequenceDataLoader(args, DATA_DIR + f"multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_val", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, args["include_frame"]).create_tf_dataset()
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
            if args["top_down_attention"]:
                print("*** Using the Baseline PredNet with attention ***")
            else:
                print("*** Using Baseline PredNet ***")

        elif args["model_choice"] == "object_centric":
            # assert args["decompose_images"], "Object-Centric PredNet requires images to be decomposed"
            from PN_models.PN_ObjectCentric import PredNet
            if args["object_representations"] and args["decompose_images"]:
                print("*** Using the Object-Centric PredNet; Decomposing & classifying inputs, and maintaining & applying object representations ***")
            elif args["decompose_images"]:
                print("*** Using the Object-Centric PredNet; Decomposing inputs ***")
        else:
            raise ValueError("Invalid model choice")
        
        inputs = keras.Input(shape=(nt, im_shape[0], im_shape[1], im_shape[2]), batch_size=batch_size)
        PN = PredNet(args, im_height=im_shape[0], im_width=im_shape[1])
        outputs = PN(inputs)
        PN = keras.Model(inputs=inputs, outputs=outputs)
        
        # Finalize model
        resos = PN.layers[-1].resolutions
        PN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss="mean_squared_error")
        print("PredNet compiled...")

        if args["object_representations"]:
            # PN.layers[-1].layers[0].layers[2].conv_vae_class.encoder.trainable = False
            # PN.layers[-1].layers[0].layers[2].conv_vae_class.decoder.trainable = False
            # PN.layers[-1].layers[0].layers[2].classifier.trainable = False
            PN.layers[1].layers[0].object_representations.sVAE.trainable = False
            PN.layers[1].layers[0].object_representations.classifier.trainable = False
            # PN.layers[1].layers[0].object_representations.seq_latent_maintainer.trainable = True
            # PN.layers[1].layers[0].object_representations.or_decoder.trainable = True
            # PN.layers[-1].layers[0].layers[2].latent_LSTM.trainable = True
            PN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss="mean_squared_error")
            print("Freezing sVAE and classifier layers")

        # "Build" the model
        data_sample = next(iter(train_dataset))
        PN(data_sample[0])


        """Weights Setup"""
        # Define where weights will be loaded/saved
        if args["model_choice"] == "baseline" and not args["top_down_attention"]:
            weights_file_name = "baseline_weights.hdf5"
        elif args["model_choice"] == "baseline" and args["top_down_attention"]:
            weights_file_name = "baseline_attention_weights.hdf5"
        elif args["model_choice"] == "object_centric" and not args["object_representations"] and not args["include_frame"]:
            weights_file_name = "objectCentric_weights.hdf5"
        elif args["model_choice"] == "object_centric" and not args["object_representations"] and args["include_frame"]:
            weights_file_name = "objectCentric_weights_includeFrame.hdf5"
        elif args["model_choice"] == "object_centric" and args["object_representations"] and not args["include_frame"]:
            weights_file_name = "objectCentric_withObjectRepresentations_weights.hdf5"
            classifier_weights_file_name = "OCPN_wOR_Classifier_weights.npz"
            vae_weights_file_name = "OCPN_wOR_ConvVAE_weights.npz"
            object_representations_weights_file =  os.path.join(WEIGHTS_DIR, "OCPN_wOR_OR_weights_32_2_16.h5")
        weights_file = os.path.join(WEIGHTS_DIR, weights_file_name)
        # Define where weights will be saved with results
        results_weights_file = os.path.join(RESULTS_SAVE_DIR, f"tensorflow_weights/iteration_{training_it+1}" + weights_file_name)

        def load_pretrained_weights():
            if args["load_outside_pretrained_classifier_weights"] and args["object_representations"]:
                if os.path.exists(os.path.join(WEIGHTS_DIR, classifier_weights_file_name)):
                    try:
                        trained_classifier_weights = np.load(os.path.join(WEIGHTS_DIR, classifier_weights_file_name), allow_pickle=True)
                        trained_classifier_weights = [trained_classifier_weights[key] for key in trained_classifier_weights.keys()]
                        un_trained_classifier = PN.layers[1].layers[0].layers[2].classifier
                        un_trained_classifier.set_weights(trained_classifier_weights)
                        print("Pre-trained Classifier weights loaded successfully")
                    except:
                        print("Pre-trained Classifier weights don't fit... better fix it")
                else:
                    print("No classifier weights found - starting training from scratch")
            if args["load_outside_pretrained_vae_weights"] and args["object_representations"]:
                if os.path.exists(os.path.join(WEIGHTS_DIR, vae_weights_file_name)):
                    try:
                        trained_vae_weights = np.load(os.path.join(WEIGHTS_DIR, vae_weights_file_name), allow_pickle=True)
                        trained_vae_weights = [trained_vae_weights[key] for key in trained_vae_weights.keys()]
                        un_trained_vae = PN.layers[1].layers[0].layers[2].conv_vae_class
                        un_trained_vae.set_weights(trained_vae_weights)
                        print("Pre-trained VAE weights loaded successfully")
                    except:
                        print("Pre-trained VAE weights don't fit... better fix it")
                else:
                    print("No VAE weights found - starting training from scratch")

        if not args["restart_training"]:
            # load previously saved weights
            print("Loading weights...")
            if os.path.exists(weights_file):
                try: 
                    PN.load_weights(weights_file, by_name=True, skip_mismatch=True)
                    print("PN model weights loaded successfully - continuing training from last epoch")
                except: 
                    print("PN model weights don't fit - restarting training from scratch")
                    if args["model_choice"] == "object_centric" and args["object_representations"]: 
                        print("but loading object representations weights and resetting stored class-object experiences (sequence latent vectors)")
                        PN.layers[1].layers[0].object_representations.load_weights(object_representations_weights_file, by_name=True, skip_mismatch=True)
                        print("Object representations weights loaded successfully")
                        PN.layers[1].layers[0].object_representations.seq_latent_maintainer.reset_stored_sequence_latent_vectors()
                        print("Stored class-object experiences (sequence latent vectors) reset")
                    # load_pretrained_weights()
            else: 
                print("No PN model weights found - starting training from scratch")
                if args["model_choice"] == "object_centric" and args["object_representations"]: 
                    print("but loading object representations weights and resetting stored class-object experiences (sequence latent vectors)")
                    PN.layers[1].layers[0].object_representations.load_weights(object_representations_weights_file, by_name=True, skip_mismatch=True)
                    print("Object representations weights loaded successfully")
                    PN.layers[1].layers[0].object_representations.seq_latent_maintainer.reset_stored_sequence_latent_vectors()
                    print("Stored class-object experiences (sequence latent vectors) reset")
                # load_pretrained_weights()
        else:
            print("Restarting training from scratch")
            if args["model_choice"] == "object_centric" and args["object_representations"]: 
                print("but loading object representations weights and resetting stored class-object experiences (sequence latent vectors)")
                PN.layers[1].layers[0].object_representations.load_weights(object_representations_weights_file, by_name=True, skip_mismatch=True)
                print("Object representations weights loaded successfully")
                PN.layers[1].layers[0].object_representations.seq_latent_maintainer.reset_stored_sequence_latent_vectors()
                print("Stored class-object experiences (sequence latent vectors) reset")
            # load_pretrained_weights()
            
        # PN.build(input_shape=(batch_size, nt) + im_shape)
        print(PN.summary())
        num_layers = len(output_channels)  # number of layers in the architecture
        print(f"{num_layers} PredNet layers with resolutions:")
        for i in reversed(range(num_layers)):
            print(f"Layer {i+1}:  {resos[i][0]} x {resos[i][1]} x {output_channels[i]}")


        """Training Setup"""
        print(f"Training iteration {training_it+1} of {args['num_iterations']}")

        def lr_schedule(epoch):
            # return 1e-4
            if (training_it+1) == 1:
                # First stage training
                if epoch == 0:
                    return args["learning_rates"][0]
                elif epoch < 2 * args["nb_epoch"] // 3:
                    return args["learning_rates"][1]
                else:
                    return args["learning_rates"][2]
            elif (training_it+1) == 2:
                # Second stage training, first iteration
                # if epoch < 2:
                #     return args["learning_rates"][1]
                if epoch < args["nb_epoch"] // 3:
                    return args["learning_rates"][2]
                elif epoch < 2 * args["nb_epoch"] // 3:
                    return args["learning_rates"][3]
                else:
                    return args["learning_rates"][4]
            else:
                # Second stage training, remaining iterations
                return args["learning_rates"][4]
        
        # class CustomLRScheduler(tf.keras.callbacks.Callback):
        #     def __init__(self, patience=3, min_delta=0.01, factor=0.5, min_lr=1e-6):
        #         super(CustomLRScheduler, self).__init__()
        #         self.patience = patience
        #         self.min_delta = min_delta
        #         self.factor = factor
        #         self.min_lr = min_lr
        #         self.val_losses = []
        #         self.new_lr = 0.001

        #     def on_epoch_end(self, epoch, lr, logs=None):
        #         current_val_loss = logs.get('val_loss')
        #         self.val_losses.append(current_val_loss)
                
        #         if len(self.val_losses) > self.patience:
        #             recent_losses = self.val_losses[-self.patience-1:-1]
        #             avg_recent_loss = np.mean(recent_losses)
        #             if (recent_losses[-1] - avg_recent_loss) / avg_recent_loss > -self.min_delta:
        #                 self.new_lr = max(self.factor * float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)), self.min_lr)
        #                 tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.new_lr)
        #                 tf.print(f"\nEpoch {epoch+1}: reducing learning rate to {self.new_lr}.")

        # custom_lr_scheduler = CustomLRScheduler(patience=3, min_delta=0.01, factor=0.5, min_lr=1e-6)
        # def pass_lr(epoch, lr):
        #     return custom_lr_scheduler.new_lr

        callbacks = [LearningRateScheduler(lr_schedule)]
        if save_model:
            if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR, exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1))
            callbacks.append(ModelCheckpoint(filepath=results_weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))
        if plot_intermediate:
            callbacks.append(IntermediateEvaluations(args, data_dirs, val_dataset, val_size, batch_size=batch_size, nt=nt, output_channels=output_channels, dataset=args["dataset"], model_choice=args["model_choice"], iteration=training_it+1))
        if tensorboard:
            callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

        
        ###################################
        ##########DEBUG MODE###############
        ###################################
        # if args["debug_model"]:
        #     @tf.function
        #     def debug_step(x, y):
        #         with tf.GradientTape() as tape:
        #             predictions = PN(x)
        #             loss = tf.reduce_mean(tf.square(predictions - y))
        #         # gradients = tape.gradient(loss, PN.trainable_variables)
        #         # PN.optimizer.apply_gradients(zip(gradients, PN.trainable_variables))
        #         # PN.save_weights(os.path.join(WEIGHTS_DIR, "debug_objectCentric_withObjectRepresentations_weights.hdf5"))
        #         print(f"Loss: {loss}")

        #     try:
        #         PN.load_weights(os.path.join(WEIGHTS_DIR, "objectCentric_withObjectRepresentations_weights.hdf5"))
        #     except:
        #         print("No debug weights found - starting training from scratch")

        #     # Iterate over the dataset
        #     for x, y in train_dataset:
        #         debug_step(x, y)
        ###################################
        ##########END DEBUG MODE###########
        ###################################
        
        history = PN.fit(train_dataset, batch_size=batch_size, steps_per_epoch=sequences_per_epoch_train, epochs=nb_epoch, callbacks=callbacks, 
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
    parser.add_argument("--sequences_per_epoch_train", type=int, default=2000, help="number of sequences per epoch for training, otherwise default to dataset size / batch size if None")
    parser.add_argument("--sequences_per_epoch_val", type=int, default=50, help="number of sequences per epoch for validation, otherwise default to validation size / batch size if None")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--nb_epoch", type=int, default=30, help="number of epochs")
    parser.add_argument("--second_stage", type=bool, default=False, help="utilize 2nd stage training data even for first iteration through dataset")

    # Model args
    parser.add_argument("--output_channels", nargs="+", type=int, default=[3, 48, 96, 192], help="output channels. Decompose turns bottom 3 channels to 12. Including original frame adds 3 channels.")
    parser.add_argument("--downscale_factor", type=int, default=4, help="downscale factor for images prior to training")
    parser.add_argument("--resize_images", type=bool, default=False, help="whether or not to downscale images prior to training")
    parser.add_argument("--decompose_images", type=bool, default=True, help="whether or not to decompose images for training")
    parser.add_argument("--include_frame", type=bool, default=False, help="whether or not to include the original frame stacked with the decomposed images for training")
    parser.add_argument("--object_representations", type=bool, default=True, help="whether or not to use object representations as input to Rep unit")
    parser.add_argument("--top_down_attention", type=bool, default=False, help="whether or not to use object representations as input to Rep unit")
    parser.add_argument("--training_split", type=float, default=1, help="proportion of data for training (only for monkaa)")

    # Training args
    parser.add_argument("--seed", type=int, default=np.random.randint(0,1000), help="random seed")
    parser.add_argument("--results_subdir", type=str, default=f"{str(datetime.now())}", help="Specify results directory")
    parser.add_argument("--restart_training", type=bool, default=True, help="whether or not to delete weights and restart")
    parser.add_argument("--reserialize_dataset", type=bool, default=True, help="reserialize dataset")
    parser.add_argument("--output_mode", type=str, default="Error", help="Error, Prediction, or Error_Images_and_Prediction. Only trains on Error.")
    parser.add_argument("--learning_rates", nargs="+", type=int, default=[5e-4, 2e-3, 1e-3, 5e-4, 1e-4], help="learning rates for each stage of training")
    parser.add_argument("--pretrain_latent_lpn", type=bool, default=False, help="this will zero out the prediction errors, and focus on the latent lstm loss")
    parser.add_argument("--load_outside_pretrained_classifier_weights", type=bool, default=True, help="load pretrained weights")
    parser.add_argument("--load_outside_pretrained_vae_weights", type=bool, default=False, help="load pretrained weights")
    parser.add_argument("--debug_model", type=bool, default=False, help="this will bypass model.fit and instead feed data through the model to debug the model")

    # Structure args
    parser.add_argument("--model_choice", type=str, default="object_centric", help="Choose which model. Options: 'baseline' or 'object_centric'")
    parser.add_argument("--system", type=str, default="laptop", help="laptop or delftblue")
    parser.add_argument("--dataset", type=str, default="SSM", help="SSM - Simple Shape Motion dataset")
    parser.add_argument("--data_subset", type=str, default="multiShape", help="provide descriptive name for results and weights")
    parser.add_argument("--dataset_size", type=int, default=100000, help="total number of images in data dir")
    parser.add_argument("--num_iterations", type=int, default=8, help="number of iterations through the dataset")
    parser.add_argument("--restart_iteration", type=int, default=2, help="Start from this iteration (# of total_#) 0 & 1 are both start. 2 skips 1st stage")
    parser.add_argument("--dataset_im_shape", nargs="+", type=int, default=[64, 64], help="output channels")
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes for object-centric model")
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
