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
load_path = "data/whole_dataset/4CH/"
masks_save_path = "data/4CH_dataset/masks/"
image_save_path = "data/4CH_dataset/image/"

#instantiate params
dim = [800,800]

data_generation_params_vent = params.Params(dim = dim,
                              experiment_name = "2CH_vent")
data_generation_params_cone = params.Params(dim = dim,
                              experiment_name = "cone")
#Load model and weights
model_vent = unet.Unet(data_generation_params_vent)
model_cone = unet.Unet(data_generation_params_cone)

for filename in os.listdir(load_path):
    #Open image with Iksnap
    subprocess.call(['itksnap',load_path + filename])   
    #Select Frames
    frames = []
    frame = 1
    while(frame != 0):
        frame = int(input("Enter frame(0 to quit): "))
        if(frame != 0):
            frames.append(frame)
    #nii to np
    image_frames,head,affine = data.load_nii_image(load_path + filename,frames)
    #predict cone masks
    cone_masks = model_cone.make_prediction(image_frames)
    cone_masks = data.thresh_mask(cone_masks,0.5)
    #apply cone masks 
    image_frames_nc = np.multiply(image_frames,cone_masks)
    #predict vent masks 
    vent_masks = model_vent.make_prediction(image_frames_nc)
    #make masks OG resolution and back to nii
    dim_padded = vent_masks.shape
    dim_og =  head.get_data_shape()
    cx,cy = int(dim_padded[1]/2),int(dim_padded[2]/2)
    predicted_masks_og_dim = np.empty([vent_masks.shape[0],dim_og[0],dim_og[1],1])
    for i in range(vent_masks.shape[0]):
        mask = data.crop_image(vent_masks[i,:,:,0],dim_padded[1:3],cx,cy,dim_og)
        mask = data.thresh_mask(mask,0.5)
        nii_ind = filename.find('.nii.gz')
        maskname = filename[:nii_ind] + '_' + str(frames[i]) + filename[nii_ind:]
        #convert np to nifti 1
        vent_masks_nb = nib.Nifti1Image(mask,affine)
        #save to disk
        nib.save(vent_masks_nb,path + masks_save_path + maskname)
        #open in itksnap
        #Itksnap -g image.nii.gz -a segmentation.nii.gz
        print("Editing mask: " + maskname + " frame: " + str(frames[i]))
        subprocess.call(['itksnap','-g',load_path + filename,'-s', masks_save_path + maskname])   
        #save image to appr folder
        subprocess.call(['cp', load_path + filename, image_save_path + filename])
        print("Saved mask: " + maskname + " and image: " + filename)



#if args.mask
#parser = argparse.ArgumentParser(description='Datapoint generation pipline')
#parser.add_argument('--mask', '-m', default=None,
#                    help='Mask refinement')
#parser.add_argument('--image', '-i', default=None,
#                    help='Image to be segmented -> mask refinement')
#
#args = parser.parse_args()
#print(args)
