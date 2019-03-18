import os
import sys
import argparse
import subprocess

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


parser = argparse.ArgumentParser(description='Datapoint generation pipline')
parser.add_argument('--mask', '-m', default=None,
                    help='Mask refinement')
parser.add_argument('--image', '-i', default=None,
                    help='Image to be segmented -> mask refinement')

args = parser.parse_args()
print(args)

#Load model and weights
json_file = open(model_path + 'model_2CH_vent.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print("loaded model from json")
model_2CH_vent = model_from_json(loaded_model_json)
model_2CH_vent.load_weights(model_path + "model_weights_2CH_vent.hdf5")
model_2CH_vent.compile(loss = f.bce_dice_loss, optimizer = optimizers.Adam(),metrics = [f.dice_loss])
model_2CH_vent.summary(line_length = 100)
print("compiled model")

if args.image
    #Open image with Iksnap
    subprocess.call(["itksnap",args.image])   
    #Select Frames
    num_frames = input("Enter number of frames: ")
    frames = []
    for i in range(num_frames):
	frames.append(input("Enter frame: "))
    #nii to np
    image_np = np.empty([len(frames),N_DIM,N_DIM,1])
    for index,frm in enumerate(frames):
        image = f._process_pathname(args.image)
        X, Y, Z, T = image.shape
        cx, cy = int(X / 2), int(Y / 2)
        crp_image = f.crop_image(image, cx, cy, N_DIM)
        pad_image_frame = pad_image[:,:,0,frames[index]-1]
        image_np[      
    #Segment Frames
    
    #np to nii

    #Open Frames in ITK-snap with image for refinement
if args.mask
