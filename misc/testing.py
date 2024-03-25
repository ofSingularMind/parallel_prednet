import hickle
import numpy as np

a = hickle.load("/home/evalexii/Documents/Thesis/code/parallel_prednet/monkaa_data/monkaa_train.hkl")

for i in range(len(a)):
    # print min and max
    print(f"min: {np.min(a[i])}, max: {np.max(a[i])}")