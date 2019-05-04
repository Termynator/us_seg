import numpy as np
import matplotlib.pyplot as plt

import data
import unet
import params

# Path to image
data_path = "data/whole_dataset/KCL_GC_020_US_2CH.nii.gz"
numpy_path = "data/single_img/"
model_path = "models/"

# Load nifti to numpy
image,head,affine = data.load_nii_image(data_path)

size = [800,800]

predict_cone_params = params.Params(dim = size,
                                    experiment_name = "cone")
model = unet.Unet(predict_cone_params)

predicted_masks = model.make_prediction(image,thresh = 0.5)

image_nc = np.multiply(image,predicted_masks)

np.save(numpy_path + "image_nc", image_nc)

