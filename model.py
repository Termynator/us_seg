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

#class Model():
#    def __init__():
#    
#    def load_architecture():
#
#    def load_weights():
#
#    def train():
#
#    def make_predcition():



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
  

def Unet(img_height, img_width, nclasses=2, filters=64):
    # down
    input_layer = Input(shape=(img_height, img_width, 1), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*4)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*4)
    conv5 = Dropout(0.5)(conv5)
    # up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*4)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    # out
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

