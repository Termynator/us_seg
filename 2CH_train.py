import os
import numpy as np

import data
import model

path = "/home/zeke/Programming/cnn/us_seg/keras/"
path_2CH = path + "data/2CH_dataset/"
model_path = path + "models/"
numpy_path_2CH = path_2CH + "numpys/"
image_path_2CH = path_2CH + "image/"
masks_path_2CH = path_2CH + "masks/"

image_ds = np.load(numpy_path_2CH + "image_ds.npy")
masks_ds = np.load(numpy_path_2CH + "masks_vent_ds.npy")

#make dynamic
size = image_ds.shape[1:3]
model = model.Model(size)

model.Model.train(model,train_ds,masks_ds)
