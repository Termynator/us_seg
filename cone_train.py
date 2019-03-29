import os
import numpy as np
import matplotlib.pyplot as plt

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
#TODO make cone ds

image_ds = np.load(numpy_path_2CH + "image_cone_ds.npy")
masks_ds = np.load(numpy_path_2CH + "masks_cone_ds.npy")

#instantiate params
name = "cone"
batch_size = 1
num_epochs = 200
steps_per = 50
num_folds = 0

params = params.Params(name,batch_size,num_epochs,steps_per,num_folds)
print(image_ds.shape)
print(masks_ds.shape)

#make dynamic
size = [800,800] #image_ds.shape[1:3]
model = unet.Unet(size,params)
model.get_Unet()
print(model.params.callbacks)
model.train(image_ds,masks_ds)
