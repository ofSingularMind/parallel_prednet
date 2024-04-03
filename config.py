# # Where Monkaa data is stored
# # If you directly download the processed data, change to the path of the data.
# DATA_DIR = '',

# # Where model weights are stored
# # If you directly download the trained weights, change to appropriate path.
# WEIGHTS_DIR = '',

# # Where results (prediction plots and evaluation file) will be saved.
# RESULTS_SAVE_DIR = '',

# # Where Tensorboard logs will be saved if you run PPN_train.py
# LOG_DIR = ''

settings = {'dirs': ''}

# Unpack with DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = settings['dirs'], for example

def update_settings(system, dataset, subdir):
    global settings
    settings_dict = {}
    settings_dict[('laptop', 'monkaa')] = {'dirs' : ['/home/evalexii/Documents/Thesis/code/parallel_prednet/monkaa_data/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/monkaa/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/monkaa_results/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/monkaa/'
                       ]
    }

    settings_dict[('delftblue', 'monkaa')] = {'dirs' : ['/scratch/aledbetter/parallel_prednet/monkaa_data/',
                          '/scratch/aledbetter/parallel_prednet/model_weights/monkaa/',
                          f'/scratch/aledbetter/parallel_prednet/monkaa_results/{subdir}/',
                          '/scratch/aledbetter/parallel_prednet/logs/monkaa/'
                          ]
    }
    settings_dict[('laptop', 'kitti')] = {'dirs' : ['/home/evalexii/Documents/Thesis/code/parallel_prednet/kitti_data/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/kitti/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/kitti_results/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/kitti/'
                       ]
    }

    settings_dict[('delftblue', 'kitti')] = {'dirs' : ['/scratch/aledbetter/parallel_prednet/kitti_data/',
                          '/scratch/aledbetter/parallel_prednet/model_weights/kitti/',
                          f'/scratch/aledbetter/parallel_prednet/kitti_results/{subdir}/',
                          '/scratch/aledbetter/parallel_prednet/logs/kitti/'
                          ]
    }

    if (system, dataset) in settings_dict:
        settings.update(settings_dict[(system, dataset)])
    else:
        raise ValueError("Invalid system / dataset. Choose ('laptop' or 'delftblue') / ('kitti' or 'monkaa').")


def get_settings():
    return settings
