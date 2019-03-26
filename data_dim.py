import os
import nibabel as nib

path = "/home/zeke/Programming/cnn/us_seg/data/whole_dataset/"
DIM_X = 0
DIM_Y = 0

for file_name in os.listdir(path):
        image_nb = nib.load(os.path.join(path,file_name))
        head = image_nb.header
        dim = head.get_data_shape()
        if (dim[0] > DIM_X):
            DIM_X = dim[0]
        if (dim[1] > DIM_Y):
            DIM_Y = dim[1]

print(DIM_X)
print(DIM_Y)

