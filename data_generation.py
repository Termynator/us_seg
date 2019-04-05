import os
import sys
import argparse
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import data
import params
import unet

path = "/home/zeke/Programming/cnn/us_seg/"
load_path = "data/whole_dataset/"
save_path = "data/data_gen/4CH_ds"
#instantiate params
dim = [800,800]

data_generation_params_vent = params.Params(dim = dim,
                              experiment_name = "2CH_vent")
data_generation_params_cone = params.Params(dim = dim,
                              experiment_name = "cone")
#Load model and weights
#model_vent = unet.Unet(data_generation_params_vent)
model_cone = unet.Unet(data_generation_params_cone)

for filename in os.listdir(load_path):
    #Open image with Iksnap
    subprocess.call(['itksnap',load_path + filename])   
    #Select Frames
    num_frames = int(input("Enter number of frames: "))
    frames = []
    for i in range(num_frames):
        frames.append(int(input("Enter frame: ")))
    #nii to np
    image_frames,head = data.load_nii_image(load_path + filename,frames)
    print(image_frames.shape)
    plt.imshow(image_frames[0,:,:,0])
    plt.show()
    #predict cone masks
    cone_masks = model_cone.make_prediction(image_frames)
    print(cone_masks.shape)
    plt.imshow(cone_masks[0,:,:,0])
    plt.show()
    #apply cone masks 
    image_frames_nc = np.multiply(image_frames,cone_masks)
    plt.imshow(image_frames_nc[0,:,:,0])
    plt.show()
    #predict vent masks 
    vent_masks = model_vent.make_prediction(image_frames_nc)
    plt.imshow(vent_masks[0,:,:,0])
    plt.show()
    #make masks OG resolution and back to nii
    dim_padded = vent_masks.shape
    dim_og =  head.get_data_shape()
    cx,cy = int(dim_padded[1]/2),int(dim_padded[2]/2)
    predicted_masks_og_dim = np.empty([vent_masks.shape[0],dim_og[0],dim_og[1],1])
    for i in range(vent_masks.shape[0]):
#        predicted_masks_og_dim[i,:,:,:] = data.crop_image(predicted_masks[i,:,:,:],cx,cy,dim_og)
        mask = data.crop_image(vent_masks[i,:,:,:],cx,cy,dim_og)
        print(mask.shape)
        mask = mask[:,:,0]
        mask = data.thresh_mask(mask,0.5)
        plt.imshow(mask)
        plt.show()
        print(filename)
        nii_ind = filename.find('.nii.gz')
        maskname = filename[:nii_ind] + '_' + str(frames[i]) + filename[nii_ind:]
        print(maskname)
        nib.save(vent_masks[i,:,:,:],save_path + maskname)
        #save to disk
        #open in itksnap
    print(predicted_masks_og_dim.shape)
    #save to disk
    #Open Frames in ITK-snap with image for refinement




#if args.mask
#parser = argparse.ArgumentParser(description='Datapoint generation pipline')
#parser.add_argument('--mask', '-m', default=None,
#                    help='Mask refinement')
#parser.add_argument('--image', '-i', default=None,
#                    help='Image to be segmented -> mask refinement')
#
#args = parser.parse_args()
#print(args)
