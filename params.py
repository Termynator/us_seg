#!/usr/bin/env/ python3
import os
import pandas
import numpy as np

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import optimizers 

class Params():

    def __init__(self,dim,experiment_name,batch_size=1,num_epochs=200,steps_per_epoch=50,num_folds=0):
        #train
        self.dim = dim
        self.path = "/home/zeke/Programming/cnn/us_seg/"
        self.model_path = self.path + "models/"
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.first_epoch = self.get_first_epoch()
        self.num_epochs = num_epochs + self.first_epoch
        self.steps_per_epoch = steps_per_epoch
        self.decay = 0.001
        self.num_drops = 1
        self.drop = 0.1
        self.init_lr = 0.001
        
        #CV
        self.num_folds = num_folds
        #optimizer
        opt = "adam"
        #stick to default params
        if (opt == "adam"):
            self.optimizer = optimizers.Adam(decay = self.decay)
        if (opt == "sgd"):
            self.optimizer = optimizers.SGD(decay = self.decay)

        #data augmentation
        self.data_gen_args = dict(samplewise_center = False,
                                featurewise_center = False,
                                samplewise_std_normalization = False,
                                featurewise_std_normalization = False,
                                rotation_range=45,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range = 0.1,
                                zoom_range=0.1,
                                vertical_flip = True,
                                horizontal_flip = True,
                                fill_mode = "nearest")

        self.data_val_args = dict(samplewise_center = False,
                                featurewise_center = False,
                                samplewise_std_normalization = False,
                                featurewise_std_normalization = False,
                                rotation_range=45,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range = 0.1,
                                zoom_range=0.1,
                                vertical_flip = True,
                                horizontal_flip = True,
                                fill_mode = "nearest")
        #callbacks
        self.final = self.model_path + self.experiment_name + '_final.hdf5'
        self.best = self.model_path + self.experiment_name + '_best.hdf5'
        self.callbacks = [CSVLogger(self.model_path + self.experiment_name + '.csv', append=True),
                          ModelCheckpoint(self.best, monitor='val_loss', verbose=2, save_best_only=True),
                          LearningRateScheduler(self.drop_decay, verbose=1)]      

    def drop_decay(self,epoch):
        epochs_drop = np.floor(self.num_epochs / (self.num_drops + 1))
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / epochs_drop))
        return lr

    def get_first_epoch(self):
        csv_file = self.model_path + self.experiment_name + '.csv'
        if os.path.isfile(csv_file):
            with open(csv_file, 'r') as f:
                first_epoch = pandas.read_csv(f).iloc[-1]['epoch'] + 1
        else:
            first_epoch = 0
        return first_epoch 
                    
