#!/usr/bin/env/ python3
import os
import pandas
import numpy as np

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import optimizers 

path = "/home/zeke/Programming/cnn/us_seg/"
model_path = path + "models/"

class Params():

    def __init__(self,exp_name,batch_size,num_epochs,steps_per_epoch,num_folds):
        #train
        self.experiment_name = exp_name
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
                                samplewise_std_normalization = False,
                                rotation_range=45,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.1)


         
        #callbacks
    def drop_decay(self,epoch):
        epochs_drop = np.floor(self.num_epochs / (self.num_drops + 1))
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / epochs_drop))
        return lr

    def get_first_epoch(self):
        csv_file = model_path + self.experiment_name + '.csv'
#        if os.path.isfile(csv_file):
#            with open(csv_file, 'r') as f:
#                first_epoch = pandas.read_csv(f).iloc[-1]['epoch'] + 1
#        else:
#            first_epoch = 0
        first_epoch = 0
        return first_epoch 
                    
