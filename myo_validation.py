import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

import data
import unet
import params

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/myo_dataset/"
model_path = path + "models/"
numpy_path = data_path + "numpys/"

image_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_myo_ds.npy")

dim = [800,800]

seed = 0
np.random.seed(seed)
num_folds = 5
fold_ind = np.arange(0,image_ds.shape[0])
np.random.shuffle(fold_ind)
folds_ind = np.array_split(fold_ind,num_folds)

#test_inds = [[25,66,85,82,44, 3,32,87,97,42,13,95,14,90,56,50,47,17,72,83]
#            ,[43,79,20,98,15,48,33,46,63,39,92, 4,62,35,73,51,22,69,94,38]
#            ,[89,68,88,31,12,49, 9,19,60,37,58, 8,45,54,84,21,52,78,61,30]  
#            ,[16,41,23,65,10,76,67,81,64,86,11,55,40,18,70,34,77, 6,24,36] 
#            ,[53, 2,59,91,71,28, 7, 0,57,93,29, 1,75,96,80,26, 5,74,27,0]]

dice = np.zeros([5,20])
for i in range(0,5):
# load model
    print("fold: " + str(i+1))
    name = "myo_cv_" + str(i+1)

# make correct data sets
    train_ind = np.concatenate(np.delete(folds_ind,i,0))
    test_ind = folds_ind[i]
    print(train_ind)
    print(test_ind)

#    test_ind = test_inds[i]

#load coresponding 5 fold cv validation set
    image_test_ds = image_ds[test_ind,:,:,:]
    masks_test_ds = masks_ds[test_ind,:,:,:]
 
    vent_val_params = params.Params(dim = dim, experiment_name = name)
    model = unet.Unet(vent_val_params)
 
    predicted_masks_cv = model.make_prediction(image_test_ds,thresh = 0.5)
    
    print(image_test_ds.shape)
    print(predicted_masks_cv.shape)

    for j in range(0,image_test_ds.shape[0]):
        dice[i,j] = unet.dice_coeff_np(masks_test_ds[j,:,:,0],predicted_masks_cv[j,:,:,0])
           
# ROC
    predicted_masks_roc = model.make_prediction(image_test_ds,thresh = None)
    fpr,tpr,thresholds = roc_curve(masks_test_ds.flatten(),predicted_masks_roc.flatten())
    plt.plot(fpr,tpr,label = "Fold " + str(i+1) + '(AOC: ' + str(auc(fpr,tpr)) + ')')
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.legend()
plt.title("Cross Validated ROC Curve: LV Myocardium Segmentation")
plt.show()

# get avg dice loss
dice_avg = np.average(dice,axis = 1)
print(dice_avg)
