import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# import data
import unet
import params

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/vent_dataset/"
model_path = path + "models/"
numpy_path = data_path + "numpys/"
image_path = data_path + "image/"
masks_path = data_path + "masks/"

# load numpys
image_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_vent_ds.npy")

print('DS')
print(image_ds.shape)
print(masks_ds.shape)

# instantiate params
name = "vent"
batch_size = 1
num_epochs = 750
steps_per = 50
num_folds = 0

dim = image_ds.shape[1:3]

vent_train_params = params.Params(dim=dim, 
                       experiment_name=name,
                       batch_size=batch_size,
                       num_epochs=num_epochs,
                       steps_per_epoch=steps_per)

image_train_ds,image_test_ds,masks_train_ds,masks_test_ds = train_test_split(image_ds,masks_ds,test_size = 0.1)

# make dynamic
model = unet.Unet(vent_train_params)
model.train(image_train_ds,masks_train_ds,image_test_ds,masks_test_ds,continue_train = False)
