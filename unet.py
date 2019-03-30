#!/usr/bin/python3

import numpy as np

import nibabel as nib
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.layers import Activation, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Input
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import params

K.set_image_data_format('channels_last')
#K.set_image_dim_ordering('th')

class Unet():
    def __init__(self,params):
        self.params = params

    def get_Unet(self):
        model = Unet_arch(self.params.dim[0],self.params.dim[1])
        model.summary()
        model.compile(loss = bce_dice_loss,
                      optimizer = optimizers.Adam(),#self.params.optimizer
                      metrics = [dice_loss])
        return model

   
    def train(self,image_trn_ds,masks_trn_ds,image_val_ds = None,masks_val_ds = None,continue_train = False):
        #params
        seed_np = 6
        seed_trn = 7
        seed_val = 8
        np.random.seed(seed_np)
        num_folds = self.params.num_folds
        batch_size = self.params.batch_size
        steps_per_epoch = self.params.steps_per_epoch
        experiment_name = self.params.experiment_name
        num_epochs = self.params.num_epochs
        
        if continue_train is False:
            model = self.get_Unet()
            print("defined new model")
        if continue_train is True:
            model = self.get_Unet()
            model.load_weights(self.params.best)
            print("loaded model from weights")

        #train data augmentation with keras image generator
        im_data_gen_args = self.params.data_gen_args
        mk_data_gen_args = self.params.data_gen_args
        
        image_datagen = ImageDataGenerator(**im_data_gen_args)
        masks_datagen = ImageDataGenerator(**mk_data_gen_args)
        
        image_generator_trn = image_datagen.flow(image_trn_ds,seed=seed_trn,batch_size=batch_size)
        masks_generator_trn = masks_datagen.flow(masks_trn_ds,seed=seed_trn,batch_size=batch_size)
        image_generator_val = image_datagen.flow(image_val_ds,seed=seed_val,batch_size=batch_size)
        masks_generator_val = masks_datagen.flow(masks_val_ds,seed=seed_val,batch_size=batch_size)

        train_generator = zip(image_generator_trn, masks_generator_trn)
        valid_generator = zip(image_generator_val, masks_generator_val)
        print("defined training and validation image generators")

#        model_path = "/home/zeke/Programming/cnn/us_seg/models/"
#        model_save_path = model_path  + "model_" + self.params.experiment_name + "_weights.hdf5"
#	#monitor should be related to dice
#        print("defined callbacks")

        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=num_epochs,
                            validation_data = valid_generator,
                            validation_steps = 5,
                            callbacks=self.params.callbacks)
        



#    def k_fold_cv():
#
    def make_prediction(self,image):
        #takes image or images and generates predicted masks
        model = self.get_Unet()
        print('defined model')
        #loads best weights
        model.load_weights(self.params.best)
        print('loaded best weights')
        prediction = np.empty_like(image)
        for i in range(image.shape[0]):
            print("Segging Image: " + str(i))
            #predictions = model.predict_generator(self.validation_gen)
            prediction[i,:,:,:] = model.predict(image[i:i+1,:,:,:])
        return prediction


def dice_coeff(y_true, y_pred):
  smooth=1
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(K.abs(y_true_f * y_pred_f))#,axis=-1
  return (2. * intersection + smooth) / (K.sum(K.square(y_true_f),-1) + K.sum(K.square(y_pred_f),-1) + smooth)
  
def dice_loss(y_true, y_pred):
  loss = 1 - dice_coeff(y_true, y_pred)
  return loss
    
def bce_dice_loss(y_true, y_pred):
  loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  return loss

def conv_block(tensor, nfilters, size = 3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size = 3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y
  

def Unet_arch(img_height, img_width, nclasses=2, filters=64):
    # down
    input_layer = Input(shape=(img_height, img_width, 1), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*2)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*4)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*4)
    conv5 = Dropout(0.5)(conv5)
    # up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*4)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*2)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    # out
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

