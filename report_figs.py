import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/vent_dataset/"
numpy_path = data_path + "numpys/"
model_path = path + "models/"
figure_path = path + "report/images/"

# load numpys
image_cn_ds = np.load(numpy_path + "image_ds.npy")
image_nc_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_vent_ds.npy")

# images for explanation
image = image_ds[0,:,:,:]
mask = masks_ds[0,:,:,:]

# 5 fold cv training progress
for i in range(0,5):
    name = "vent_cv_" + str(i+1)
    
    training_df = pd.read_csv(model_path + name + ".csv")

    epochs = training_df.iloc[:,0]
    trn_loss = training_df.iloc[:,1]
    val_loss = training_df.iloc[:,4]

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Fold ' + str(i+1))
    plt.legend()

    plt.plot(epochs, trn_loss, val_loss,)

    plt.savefig('fold_' + str(i+1) + '.png')
