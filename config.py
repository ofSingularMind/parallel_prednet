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

def update_settings(system, dataset, data_subset, subdir):
    global settings
    settings_dict = {}
    settings_dict[('laptop', 'monkaa')] = {'dirs' : ['/home/evalexii/Documents/Thesis/code/parallel_prednet/data/monkaa_data/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/monkaa/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/monkaa_results/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/monkaa/'
                       ]
    }

    settings_dict[('laptop', 'various')] = {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/various/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/various_results/{data_subset}/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/various/'
                       ]
    }

    settings_dict[('laptop', 'all_rolling')] = {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/all_rolling/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/all_rolling_results/{data_subset}/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/all_rolling/'
                       ]
    }

    settings_dict[('laptop', 'rolling_square')] = {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/rolling_square/frames/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/rolling_square/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/rolling_square_results/{data_subset}/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/rolling_square/'
                       ]
    }

    settings_dict[('laptop', 'rolling_circle')] = {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/rolling_circle/frames/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/rolling_circle/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/rolling_circle_results/{data_subset}/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/rolling_circle/'
                       ]
    }

    settings_dict[('laptop', 'driving')] = {'dirs' : ['/home/evalexii/Documents/Thesis/code/parallel_prednet/data/driving_data/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/driving/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/driving_results/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/driving/'
                       ]
    }

    settings_dict[('laptop', 'kitti')] = {'dirs' : ['/home/evalexii/Documents/Thesis/code/parallel_prednet/data/kitti_data/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/kitti/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/kitti_results/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/kitti/'
                       ]
    }

    settings_dict[('delftblue', 'monkaa')] = {'dirs' : ['/scratch/aledbetter/parallel_prednet/data/monkaa_data/',
                          '/scratch/aledbetter/parallel_prednet/model_weights/monkaa/',
                          f'/scratch/aledbetter/parallel_prednet/results/monkaa_results/{subdir}/',
                          '/scratch/aledbetter/parallel_prednet/logs/monkaa/'
                          ]
    }

    settings_dict[('delftblue', 'driving')] = {'dirs' : ['/scratch/aledbetter/parallel_prednet/data/driving_data/',
                          '/scratch/aledbetter/parallel_prednet/model_weights/driving/',
                          f'/scratch/aledbetter/parallel_prednet/results/driving_results/{subdir}/',
                          '/scratch/aledbetter/parallel_prednet/logs/driving/'
                          ]
    }

    settings_dict[('delftblue', 'kitti')] = {'dirs' : ['/scratch/aledbetter/parallel_prednet/data/kitti_data/',
                          '/scratch/aledbetter/parallel_prednet/model_weights/kitti/',
                          f'/scratch/aledbetter/parallel_prednet/results/kitti_results/{subdir}/',
                          '/scratch/aledbetter/parallel_prednet/logs/kitti/'
                          ]
    }

    if (system, dataset) in settings_dict:
        settings.update(settings_dict[(system, dataset)])
    else:
        try:
            settings.update(
                {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/{dataset}/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/{dataset}_results/{data_subset}/{subdir}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/{dataset}/'
                       ]
                }
            )
        except:
            raise ValueError("Invalid system / dataset. Choose ('laptop' or 'delftblue') / ('kitti' or 'monkaa').")


def get_settings():
    return settings
