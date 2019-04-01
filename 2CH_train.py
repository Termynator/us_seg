import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import data
import unet
import params

path = "/home/zeke/Programming/cnn/us_seg/"
path_2CH = path + "data/2CH_dataset/"
model_path = path + "models/"
numpy_path_2CH = path_2CH + "numpys/"
image_path_2CH = path_2CH + "image/"
masks_path_2CH = path_2CH + "masks/"

#load numpys

image_ds = np.load(numpy_path_2CH + "image_nc_ds.npy")
masks_ds = np.load(numpy_path_2CH + "masks_vent_ds.npy")

#instantiate params
name = "2CH_vent"
batch_size = 1
num_epochs = 250
steps_per = 50
num_folds = 0

dim = image_ds.shape[1:3]

cone_train_params = params.Params(dim = dim, 
                       experiment_name = name,
                       batch_size = batch_size,
                       num_epochs = num_epochs,
                       steps_per_epoch = steps_per)

image_train_ds,image_test_ds,masks_train_ds,masks_test_ds = train_test_split(image_ds,masks_ds,test_size = 0.1)

#make dynamic
model = unet.Unet(cone_train_params)
model.get_Unet()
print(model.params.callbacks)
model.train(image_train_ds,masks_train_ds,image_test_ds,masks_test_ds,continue_train = True)
#model.train(image_ds,masks_ds,continue_train = True)
