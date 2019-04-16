import os
import sys
import argparse
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import data

path = "/home/zeke/Programming/cnn/us_seg/"
load_path = "data/whole_dataset/"
masks_save_path = "data/myo_dataset/masks/"
image_save_path = "data/myo_dataset/image/"

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
    if(len(frames) >= 1):
        #nii to np
        image_frames,head,affine = data.load_nii_image(load_path + filename,frames)
        for i in range(len(frames)):
            image_dim = head.get_data_shape()
            mask_dim = image_dim[0:-1]
            mask = np.zeros(mask_dim) 
            print(mask.shape)
            nii_ind = filename.find('.nii.gz')
            if(frames[i] < 10):
                maskname = filename[:nii_ind] + '_0' + str(frames[i]) + filename[nii_ind:]
            else:
                 maskname = filename[:nii_ind] + '_' + str(frames[i]) + filename[nii_ind:]
            #convert np to nifti 1
            vent_masks_nb = nib.Nifti1Image(mask,affine)
            #save to disk and move image to correct dir
            subprocess.call(['mv', load_path + filename, image_save_path + filename])
            nib.save(vent_masks_nb,path + masks_save_path + maskname)
            #open in itksnap
            #Itksnap -g image.nii.gz -a segmentation.nii.gz
            print("Editing mask: " + maskname + " frame: " + str(frames[i]))
            subprocess.call(['itksnap','-g',image_save_path + filename,'-s',masks_save_path + maskname])   
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
