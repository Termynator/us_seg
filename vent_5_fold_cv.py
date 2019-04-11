import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as plt


path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/vent_dataset/"
model_path = path + "models/"
numpy_path = data_path + "numpys/"
image_path = data_path + "image/"
masks_path = data_path + "masks/"

# load numpys
image_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_vent_ds.npy")

# instatiate params
batch_size = 1
num_epochs = 750#up to change
steps_per = 50
num_folds = 0# remove this from params please!

# kfold stuff
kfold = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)
cv_scores = []
fold_ind = np.arange(0,image_ds.shape[0])
np.random.shuffle(fold_ind)
folds_ind = np.array_split(fold_ind,num_folds)

for i in range (0,len(folds_ind)):
  print("fold: " + str(i+1))  
  name = "vent_cv_" + str(i+1)

  vent_train_params = params.Params(dim = dim,
                       experiment_name = name,
                       batch_size = batch_size,
                       num_epochs = num_epochs,

  model = unet.Unet(vent_train_params)

  train_ind = np.concatenate(np.delete(folds_ind,i))
  test_ind = folds_ind[i]
  
  image_train = image_ds[train_ind,:,:,:]
  masks_train = masks_ds[train_ind,:,:,:]
  image_test = image_ds[test_ind,:,:,:]
  masks_test = masks_ds[test_ind,:,:,:]
  
  model.train(model.train(image_train_ds,masks_train_ds,image_test_ds,masks_test_ds)
  scores = model.evaluate(image_test, masks_test, verbose = 1)
  cv_scores.append(scores)
  print(scores)

print("%.2f% (+/- %.2f%)" % (np.mean(cv_scores), np.std(cv_scores)))
#print(cv_scores)
#Use this mask specificly to demonstrate post processing need on predicted masks.
#mask_predict = model_2CH_vent.predict(image_ds[1:2,:,:,:]) 
#
#plt.pyplot.imshow(mask_predict[0,:,:,0])
#plt.pyplot.colorbar()
#print(mask_predict[1,:,:,0].shape)
#
#plt.pyplot.imshow(image_ds[1,:,:,0])

