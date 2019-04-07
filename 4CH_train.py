import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import data
import unet
import params

path = "/home/zeke/Programming/cnn/us_seg/"
path_4CH = path + "data/4CH_dataset/"
model_path = path + "models/"
numpy_path_4CH = path_4CH + "numpys/"
image_path_4CH = path_4CH + "image/"
masks_path_4CH = path_4CH + "masks/"

#load numpys

image_ds = np.load(numpy_path_4CH + "image_nc_ds.npy")
masks_ds = np.load(numpy_path_4CH + "masks_vent_ds.npy")

for i in range(image_ds.shape[0]):
    plt.imshow(image_ds[i,:,:,0])
    plt.show()

#instantiate params
name = "4CH_vent"
batch_size = 1
num_epochs = 500
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
model.train(image_train_ds,masks_train_ds,image_test_ds,masks_test_ds,continue_train = False)
#model.train(image_ds,masks_ds,continue_train = True)
