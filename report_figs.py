import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import unet
import params

path = "/home/zeke/Programming/cnn/us_seg/"
data_path = path + "data/vent_dataset/"
numpy_path = data_path + "numpys/"
model_path = path + "models/"
figure_path = path + "report/images/"

# load numpys
image_cn_ds = np.load(numpy_path + "image_vent_ds.npy")
image_nc_ds = np.load(numpy_path + "image_nc_ds.npy")
masks_ds = np.load(numpy_path + "masks_vent_ds.npy")

# images for explanation
image_cone = image_cn_ds[0,:,:,:]
image_no_cone = image_nc_ds[0,:,:,:]
mask = masks_ds[0,:,:,:]

# load model for lv seg
predict_image_params = params.Params(dim = [800,800],
                                     experiment_name = "vent_cv_1")
model = unet.Unet(predict_image_params)
predicted_vents = model.make_prediction(image)
num_image_frames = image.shape[0]

# ROC

# image for tracking LVV though time
image = np.load(path + "data/single_img/image_nc.npy")

# LV vol through time
vent_size = np.empty([num_image_frames])
print(vent_size.shape)

for i in range(0,num_image_frames):
    vent_size[i] = np.count_nonzero(predicted_vents[i,:,:,:])

#need to actually generate volume instead of pix count

print(vent_size)
fig = plt.plot(vent_size)
plt.xlabel("Frame")
plt.ylabel("Volume")
plt.title("Evolution of LV Volume Through Time")
plt.show()
plt.savefig(figure_path + 'fold_' + str(i+1) + '.png')

# 5 fold cv training progress
for i in range(0,5):
    name = "vent_cv_" + str(i+1)
    
    #cross val loss
    fold_eval_params = params.Params(dim = [800,800],
                             experiment_name = "vent_cv_" + str(i+1))
    model = unet.Unet(fold_eval_params)

    # train hist
    training_df = pd.read_csv(model_path + name + ".csv")

    epochs = training_df.iloc[:,0]
    trn_loss = training_df.iloc[:,1]
    val_loss = training_df.iloc[:,4]

    fig, ax = plt.subplots()
    ax.plot(epochs,trn_loss,label='training loss')
    ax.plot(epochs,val_loss,label='validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Fold ' + str(i+1))
    plt.legend()

#    plt.show()
    plt.savefig(figure_path + 'fold_' + str(i+1) + '.png')
