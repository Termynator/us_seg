import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as plt

import functions as f

from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

path = "/home/zeke/Programming/CNN/US_Segmentation_Keras/"
path_2CH = path + "data/2CH_dataset/"
model_path = path + "models/"
numpy_path_2CH = path_2CH + "numpys/"
image_path_2CH = path_2CH + "image/"
masks_path_2CH = path_2CH + "masks/"

image_ds = np.load(numpy_path_2CH + "image_nc_ds.npy")
masks_ds = np.load(numpy_path_2CH + "masks_vent_ds.npy")

#json_file = open(model_path + 'model_2CH_vent.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#print("loaded model from json")
#model_2CH_vent = model_from_json(loaded_model_json)
#model_2CH_vent.load_weights(model_path + "model_weights_2CH_vent.hdf5")
#model_2CH_vent.compile(loss = f.bce_dice_loss, optimizer = optimizers.Adam(),metrics = [f.dice_loss])
#model_2CH_vent.summary(line_length = 100)
#print("compiled model")

seed = 7
np.random.seed(seed)
num_folds = 5 
batch_size = 2 
X_DIM = len(image_ds[1])
Y_DIM = len(image_ds[2])

kfold = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)
cv_scores = []

#data augmentation with keras image generator
im_data_gen_args = dict(samplewise_center = False,
                        samplewise_std_normalization = False,
                        rotation_range=45,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1)

mk_vn_data_gen_args = dict(samplewise_center = False,
                           samplewise_std_normalization = False,
                           rotation_range=45,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1)

#, preprocessing_function = aug.keras_preprocess_func()
image_datagen = ImageDataGenerator(**im_data_gen_args)
masks_vent_datagen = ImageDataGenerator(**mk_vn_data_gen_args)

image_no_cone_generator = image_datagen.flow(image_ds,seed=seed,batch_size=batch_size)
masks_vent_generator = masks_vent_datagen.flow(masks_ds,seed=seed,batch_size=batch_size)
vent_generator = zip(image_no_cone_generator, masks_vent_generator)
print("defined image generator")

model_ventricle_save_path = model_path  + "model_weights_2CH_vent.hdf5"
checkpoint = ModelCheckpoint(model_ventricle_save_path, monitor = 'val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_vent = [checkpoint]
callbacks = callbacks_vent

fold_ind = np.arange(0,image_ds.shape[0])
np.random.shuffle(fold_ind)
folds_ind = np.array_split(fold_ind,num_folds)

for i in range (0,len(folds_ind)):
  print("fold: " + str(i+1))  
  
  train_ind = np.concatenate(np.delete(folds_ind,i))
  test_ind = folds_ind[i]
  
  image_train = image_ds[train_ind,:,:,:]
  masks_train = masks_ds[train_ind,:,:,:]
  image_test = image_ds[test_ind,:,:,:]
  masks_test = masks_ds[test_ind,:,:,:]
  
  image_no_cone_generator = image_datagen.flow(image_train,seed=seed,batch_size=batch_size)
  masks_vent_generator = masks_vent_datagen.flow(masks_train,seed=seed,batch_size=batch_size)
  vent_generator = zip(image_no_cone_generator, masks_vent_generator)


  model_2CH_vent = f.Unet(X_DIM,Y_DIM)
  model_2CH_vent.compile(loss = f.bce_dice_loss, optimizer = optimizers.Adam(),metrics = [f.dice_loss])
# model_2CH_vent.summary()
  model_2CH_vent.fit_generator(vent_generator,steps_per_epoch=50,epochs=200,callbacks=callbacks)
  
  scores = model_2CH_vent.evaluate(image_test, masks_test, verbose = 1)
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

