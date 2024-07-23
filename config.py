# # Where Monkaa data is stored
# # If you directly download the processed data, change to the path of the data.
# DATA_DIR = '',

# # Where model weights are stored
# # If you directly download the trained weights, change to appropriate path.
# WEIGHTS_DIR = '',

# # Where results (prediction plots and evaluation file) will be saved.
# RESULTS_SAVE_DIR = '',

# # Where Tensorboard logs will be saved if you run PN_train.py
# LOG_DIR = ''

settings = {'dirs': ''}

# Unpack with DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = settings['dirs'], for example

def update_settings(system, dataset, data_subset, subdir):
    global settings
    settings_dict = {}

    settings_dict[('laptop', 'SSM')] = {'dirs' : [f'/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/{data_subset}/',
                       f'/home/evalexii/Documents/Thesis/code/parallel_prednet/results/SSM_results/{data_subset}/{subdir}/',
                       '/home/evalexii/Documents/Thesis/code/parallel_prednet/logs/SSM/'
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
            raise ValueError("Invalid system / dataset. Choose ('laptop') / ('SSM').")


def get_settings():
    return settings
