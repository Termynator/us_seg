#!/usr/bin/env/ python3

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import optimizers 

class Params():

    def __init__(self,exp_name,batch_size,num_epochs,steps_per_epoch,num_folds):
        #train
        self.experiment_name = exp_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        
        #CV
        self.num_folds = num_folds
        #optimizer
        opt = "adam"
        self.decay = 0.001
        self.init_lr = 0.001
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
        #self.final = self.experiment_name + '_final.hdf5'
        self.best =  self.experiment_name + '_best.hdf5'
        self.callbacks = [CSVLogger( self.experiment_name + '.csv', append=True),
                          ModelCheckpoint(self.best, monitor='val_loss', verbose=2, save_best_only=True),
                          LearningRateScheduler(self.drop_decay, verbose=1)]

    def drop_decay(self,epoch):
        epochs_drop = np.floor(self.num_epochs / (self.ndrops + 1))
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / epochs_drop))
        return lr
                    
