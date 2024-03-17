def main(args):

    import os
    import warnings

    # Suppress warnings
    warnings.filterwarnings('ignore')
    # or '2' to filter out INFO messages too
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    import shutil
    import keras
    from keras import backend as K
    from keras import layers
    from data_utils import IntermediateEvaluations, create_dataset_from_serialized_generator, config_gpus
    from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
    from PPN import ParaPredNet
    import matplotlib.pyplot as plt


    # use mixed precision for faster runtimes and lower memory usage
    # keras.mixed_precision.set_global_policy("mixed_float16")
    config_gpus()

    # if results directory already exists, then delete it
    if os.path.exists(RESULTS_SAVE_DIR):
        shutil.rmtree(RESULTS_SAVE_DIR)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    save_model = True  # if weights will be saved
    plot_intermediate = True  # if the intermediate model predictions will be plotted
    tensorboard = True  # if the Tensorboard callback will be used
    # where weights are loaded prior to training
    weights_checkpoint_file = os.path.join(
        WEIGHTS_DIR, 'para_prednet_monkaa_weights.hdf5')
    # where weights will be saved
    weights_file = os.path.join(
        WEIGHTS_DIR, 'para_prednet_monkaa_weights.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'para_prednet_monkaa_model_ALEX.json')
    if os.path.exists(weights_file): os.remove(weights_file)  # Careful: this will delete the weights file

    # Training data
    pfm_paths = []
    pfm_paths.append(DATA_DIR + 'disparity/family_x2/left/')
    pfm_paths.append(DATA_DIR + 'material_index/family_x2/left/')
    pfm_paths.append(DATA_DIR + 'object_index/family_x2/left/')
    pfm_paths.append(DATA_DIR + 'optical_flow/family_x2/into_future/left/')
    pgm_paths = []
    pgm_paths.append(DATA_DIR + 'motion_boundaries/family_x2/into_future/left/')
    png_paths = []
    png_paths.append(DATA_DIR + 'frames_cleanpass/family_x2/left')
    num_sources = len(pfm_paths) + len(pgm_paths) + len(png_paths)

    # Training parameters
    nt = args["nt"]  # number of time steps
    nb_epoch = args["nb_epoch"]  # 150
    batch_size = args["batch_size"]  # 4
    sequences_per_epoch_train = args["sequences_per_epoch_train"]  # 500
    sequences_per_epoch_val = args["sequences_per_epoch_val"]  # 500
    assert sequences_per_epoch_train is None or type(sequences_per_epoch_train) == int # this will override the default of (dataset size / batch size)
    assert sequences_per_epoch_val is None or type(sequences_per_epoch_val) == int # this will override the default of (dataset size / batch size)
    # N_seq_val = 20  # number of sequences to use for validation
    num_P_CNN = args["num_P_CNN"]
    num_R_CLSTM = args["num_R_CLSTM"]
    output_channels = args["output_channels"]
    original_im_shape = args["original_im_shape"]
    downscale_factor = args["downscale_factor"]
    im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, 3)

    #  Create and split dataset
    dataset, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode='Error',
                                                            im_height=im_shape[0], im_width=im_shape[1],
                                                            batch_size=batch_size, nt=nt, reserialize=False, shuffle=True, resize=True)

    ts = args["train_proportion"]
    vs = (1 - ts) / 2
    train_size = int(ts * length)
    val_size = int(vs * length)
    test_size = int(vs * length)
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    train_dataset = dataset.take(train_size).repeat()
    temp_dataset = dataset.skip(train_size)
    val_dataset = temp_dataset.skip(val_size).repeat()
    test_dataset = temp_dataset.take(test_size).repeat()

    inputs = (
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 1)),
        keras.Input(shape=(nt, im_shape[0], im_shape[1], 3)),
    )
    PPN = ParaPredNet(batch_size=batch_size, nt=nt, im_height=im_shape[0], im_width=im_shape[1], num_P_CNN=num_P_CNN, num_R_CLSTM=num_R_CLSTM, output_channels=output_channels)
    resos = PPN.resolutions
    outputs = PPN(inputs)
    PPN = keras.Model(inputs=inputs, outputs=outputs)
    PPN.compile(optimizer='adam', loss='mean_squared_error')
    print("ParaPredNet compiled...")
    print(PPN.summary())

    num_layers = len(output_channels)  # number of layers in the architecture
    print(f"Top layer resolution: {resos[-1][0]} x {resos[-1][1]}")

    # load previously saved weights
    if os.path.exists(weights_checkpoint_file):
        PPN.load_weights(weights_checkpoint_file)


    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    def lr_schedule(epoch): return 0.01 if epoch < 5 else 0.0001


    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
        callbacks.append(ModelCheckpoint(filepath=weights_file,
                        monitor='val_loss', save_best_only=True, save_weights_only=True))
    if plot_intermediate:
        callbacks.append(IntermediateEvaluations(test_dataset, batch_size=batch_size,
                        nt=nt, output_channels=output_channels))
    if tensorboard:
        callbacks.append(TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

    history = PPN.fit(train_dataset, 
                    steps_per_epoch=train_size / batch_size if sequences_per_epoch_train is None else sequences_per_epoch_train, 
                    epochs=nb_epoch, 
                    callbacks=callbacks,
                    validation_data=val_dataset, 
                    validation_steps=val_size / batch_size if sequences_per_epoch_val is None else sequences_per_epoch_val)


if __name__ == "__main__":
    import argparse
    from config import update_settings, get_settings
    import numpy as np

    parser = argparse.ArgumentParser(description="PPN") # Training parameters

    parser.add_argument("--nt", type=int, default=10, help="sequence length")
    parser.add_argument("--nb_epoch", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--sequences_per_epoch_train", type=int, default=10, help="number of sequences per epoch for training")
    parser.add_argument("--sequences_per_epoch_val", type=int, default=5, help="number of sequences per epoch for validation")
    parser.add_argument("--num_P_CNN", type=int, default=2, help="number of parallel CNNs")
    parser.add_argument("--num_R_CLSTM", type=int, default=2, help="number of recurrent CLSTMs")
    parser.add_argument("--output_channels", nargs='+', type=int, default=[3, 12], help="output channels")
    parser.add_argument("--original_im_shape", nargs='+', type=int, default=(540, 960, 3), help="original image shape")
    parser.add_argument("--downscale_factor", type=int, default=4, help="downscale factor")
    parser.add_argument("--train_proportion", type=float, default=0.7, help="downscale factor")
    
    parser.add_argument("--seed", type=int, default=np.random.default_rng().integers(0,9999), help="random seed")

    parser.add_argument("--system", type=str, default="laptop", help="laptop or delftblue")
    parser.add_argument("--dataset", type=str, default="monkaa", help="monkaa or kitti")

    args = parser.parse_args().__dict__

    update_settings(args["system"], args["dataset"])
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = get_settings()['dirs']

    main(args)