import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import params
import unet
import data

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/myo_dataset/"
model_path = path + "models/"
numpy_path = data_path + "numpys/"

# load numpys
image_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_myo_ds.npy")

# instatiate params
batch_size = 1
num_epochs = 300 
steps_per = 40 
dim = [800,800]

# kfold stuff
seed = 0
np.random.seed(seed)
num_folds = 5# remove this from params please!
cv_scores = []
fold_ind = np.arange(0,image_ds.shape[0])
np.random.shuffle(fold_ind)
folds_ind = np.array_split(fold_ind,num_folds)

for i in range (0,num_folds):
    print("fold: " + str(i+1))  
    name = "myo_cv_" + str(i+1)
    
    myo_train_params = params.Params(dim = dim,
                         experiment_name = name,
                         batch_size = batch_size,
                         num_epochs = num_epochs,
                         steps_per_epoch = steps_per)
    
    model = unet.Unet(myo_train_params)
    train_ind = np.concatenate(np.delete(folds_ind,i,0))
    test_ind = folds_ind[i]
    print(train_ind)
    print(test_ind)
    
    image_train_ds = image_ds[train_ind,:,:,:]
    masks_train_ds = masks_ds[train_ind,:,:,:]
    image_test_ds = image_ds[test_ind,:,:,:]
    masks_test_ds = masks_ds[test_ind,:,:,:]

    #for i in range(0,image_train_ds.shape[0]):
    #   plt.subplot(1,2,1)
    #   plt.imshow(image_train_ds[i,:,:,0])
    #   plt.subplot(1,2,2)
    #   plt.imshow(masks_train_ds[i,:,:,0])
    #   plt.show()

    #for i in range(0,image_test_ds.shape[0]):
    #   plt.subplot(1,2,1)
    #   plt.imshow(image_test_ds[i,:,:,0])
    #   plt.subplot(1,2,2)
    #   plt.imshow(masks_test_ds[i,:,:,0])
    #   plt.show()
    
    model.train(image_train_ds,masks_train_ds,image_test_ds,masks_test_ds)
    #validate in other script 
