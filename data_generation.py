import os
import sys
import argparse
import subprocess

import pandas as pd
import nibabel as nib
import matplotlib as plt

import data
import params
import unet

path = "/home/zeke/Programming/cnn/us_seg/"
data_gen = path + "data_gen/"

#instantiate params
name = "2CH_vent"

2CH_data_generation_params = params.Params(dim = dim,
                              experiment_name = name)
#Load model and weights
model = unet.Unet(2CH_data_generation_params)
model.load_weights()

for filename in os.listdir(data_gen):
    if filename.endswith(".nii.gz")
        print(os.path.join(directory, filename))
	#Open image with Iksnap
	subprocess.call(["itksnap",filename])   
	#Select Frames
	num_frames = input("Enter number of frames: ")
	frames = []
	for i in range(num_frames):
	    frames.append(input("Enter frame: "))
	#nii to np
	data.load_nii_image(filename,frames)
	#np to nii

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
