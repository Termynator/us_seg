import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/vent_dataset/"
numpy_path = data_path + "numpys/"
model_path = path + "models/"

# load numpys
image_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_vent_ds.npy")

image = image_ds[0,:,:,:]
mask = masks_ds[0,:,:,:]

experiment_name = "vent"

training_df = pd.read_csv(model_path + experiment_name + ".csv")

epochs = training_df.iloc[:,0]
loss = training_df.iloc[:,1]
val_loss = training_df.iloc[:,4]


plt.plot(epochs,loss,val_loss)
plt.show()
