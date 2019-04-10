import os
import subprocess
import glob
import matplotlib.pyplot as plt

#Visual inspection/editing of entire dataset

path = "/home/zeke/Programming/cnn/us_seg/"
masks_path = "data/vent_dataset/masks/"
image_path = "data/vent_dataset/image/"

for image_name in os.listdir(image_path):
    # from filename get all associated masks
    # 2CH / 4CH filenames can get 2CH2 / 4CH2 masks
    image_name_no_suffix = image_name[:-7]
    for mask_name in glob.glob(masks_path + image_name_no_suffix + '*'):
        print('Image: ' + image_name + '   ' + 'Mask: ' + mask_name)
        subprocess.call(['itksnap','-g',image_path + image_name,'-s',mask_name])
