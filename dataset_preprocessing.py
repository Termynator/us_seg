import numpy as np

import data
import unet

# Paths to datasets
path_2CH = "data/2CH_dataset/"
numpy_path_2CH = path_2CH + "numpys/"
#load nifties into numpys
#image_cone_ds,image_vent_ds,masks_cone_ds,masks_vent_ds,img_headers = data.load_nii(path_2CH)

#save numpys in relavent dirs
np.save(numpy_path_2CH + "image_cone_ds", image_cone_ds)
np.save(numpy_path_2CH + "iamge_vent_ds", image_vent_ds)
np.save(numpy_path_2CH + "masks_cone_ds", masks_cone_ds)
np.save(numpy_path_2CH + "masks_vent_ds", masks_vent_ds)

size = image_cone_ds.shape[1:3]
print(size)

np.save(numpy_path_2CH + "image_nc_ds", image_nc_ds)































# pad_mask[pad_mask == 1] = 0 # vent mask
#     if(np.amax(pad_mask[:,:,0] == 2.0)): # only if there are multiple masks 
#         pad_mask[pad_mask == 1] = 0    # decide to seg cone instead of ventricle
     
# max_val = np.amax(pad_mask[:,:,0])




#  image = f._process_pathname(image_path_2CH + img)
#  X, Y, Z, T = image.shape
#  cx, cy = int(X / 2), int(Y / 2)
#  pad_image = f.pad_image(image, cx, cy, N_DIM)
#  image_ds[index,:,:,0] = pad_image_frame 
#  index = index + 1

#  print('Analysing: ' + msk)
#  mask = f._process_pathname(masks_path_2CH + msk)
#  X, Y, Z = mask.shape
#  cx, cy = int(X / 2), int(Y / 2)
#  pad_mask = f.pad_image(mask, cx, cy, N_DIM)
#  pad_mask[pad_mask == 2] = 1 # decide to seg cone instead of ventricle
#  max_val = np.amax(pad_mask[:,:,0])
#  masks_cone_ds[index,:,:,0] = pad_mask[:,:,0]/max_val


#import os
#import numpy as np
#import pandas as pd
##import nibabel as nib
#import matplotlib as plt
#import keras.backend as K
#
#from keras import optimizers
##from keras.losses import binary_crossentropy
#from keras.models import model_from_json
#
#import functions as f
#
##store all filenames in lists as well as frames in images that masks were aquired on
#path = "/home/zeke/Programming/CNN/US_Segmentation/"
#path_2CH = path + "data/2CH_dataset/"
#model_path = path + "models/"
#numpy_path_2CH = path_2CH + "numpys/"
#image_path_2CH = path_2CH + "image/"
#masks_path_2CH = path_2CH + "masks/"
#
#masks_filenames_2CH = pd.read_csv(os.path.join(path_2CH, 'masks_filenames_2CH.csv'))
#masks_filenames_2CH = np.squeeze(np.asarray(masks_filenames_2CH))
#image_filenames_2CH = []
#frames = []
#
#for mask in masks_filenames_2CH:
#  image_filenames_2CH.append(mask[:-10] + mask[-7:])
#  frames.append(int(mask[-9:-7]))
#
#
#N_DIM = 512
#
##want to resize ims and also get specific frames that masks refer to
##do masks and get frame numbers
#image_ds = np.empty([len(image_filenames_2CH),N_DIM,N_DIM,1]) #need shape [num ims,channels,x,y]
#index = 0
#
#for img in image_filenames_2CH:
#  print('Analysing: ' + img)
#  image = f._process_pathname(image_path_2CH + img)
#  X, Y, Z, T = image.shape
#  cx, cy = int(X / 2), int(Y / 2)
#  pad_image = f.pad_image(image, cx, cy, N_DIM)
#  pad_image_frame = pad_image[:,:,0,frames[index]-1]
#  image_ds[index,:,:,0] = pad_image_frame 
#  index = index + 1

#masks_cone_ds = np.empty([len(masks_filenames_2CH),N_DIM,N_DIM,1])
#index = 0
#
#for msk in masks_filenames_2CH:

#  print('Analysing: ' + msk)
#  mask = f._process_pathname(masks_path_2CH + msk)
#  X, Y, Z = mask.shape
#  cx, cy = int(X / 2), int(Y / 2)
#  pad_mask = f.pad_image(mask, cx, cy, N_DIM)
#  pad_mask[pad_mask == 2] = 1 # decide to seg cone instead of ventricle
#  max_val = np.amax(pad_mask[:,:,0])
#  masks_cone_ds[index,:,:,0] = pad_mask[:,:,0]/max_val
#  index = index + 1
#masks_vent_ds = np.empty([len(masks_filenames_2CH),N_DIM,N_DIM,1])
#from scipy import stats
#index = 0
#for msk in masks_filenames_2CH:
#  print('Analysing: ' + msk)
#  mask = f._process_pathname(masks_path_2CH + msk)
#  X, Y, Z = mask.shape
#  cx, cy = int(X / 2), int(Y / 2)
#  pad_mask = f.pad_image(mask, cx, cy, N_DIM)
#   
#  if(np.amax(pad_mask[:,:,0] == 2.0)): # only if there are multiple masks 
#      pad_mask[pad_mask == 1] = 0    # decide to seg cone instead of ventricle
#     
#  max_val = np.amax(pad_mask[:,:,0])
#  masks_vent_ds[index,:,:,0] = pad_mask[:,:,0]/max_val
#  index = index + 1
#    
#
#json_file = open(model_path + 'model_cone.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model_cone = model_from_json(loaded_model_json)
## load weights into new model
#model_cone.load_weights(model_path + "model_weights_cone.hdf5")
#print("Loaded cone model from disk")
#model_cone.compile(loss = f.bce_dice_loss, optimizer = optimizers.Adam(),metrics = [f.dice_loss])#metrics = dice_loss
#
#image_nc_ds = np.empty([image_ds.shape[0],N_DIM,N_DIM,1])
#index = 0
#for i in range(image_ds.shape[0]):
#    print('Segmenting Image: ' + str(i))
#    image = model_cone.predict(image_ds[i:i+1,:,:,:])
#    image = np.multiply(image,image_ds[i,:,:,:])
#    image_nc_ds[i,:,:,:] = image
#    index = index + 1
#
#print(image_ds.shape)
#print(image_nc_ds.shape)
#print(masks_cone_ds.shape)
#print(masks_vent_ds.shape)
#
#np.save(numpy_path_2CH + "image_ds", image_ds)
#np.save(numpy_path_2CH + "image_nc_ds", image_nc_ds)
#np.save(numpy_path_2CH + "masks_cone_ds", masks_cone_ds)
#np.save(numpy_path_2CH + "masks_vent_ds", masks_vent_ds)
