#!/usr/bin/env/ python3

class Parameters():

    def __init__(self,exp_name,dim)
        self.experiment_name = exp_name
        self.image_dim = dim

        #train
        self.batch_size = 1
        self.num_epochs = n_epochs
        
        #data augmentation
        self.featurewise_init = False
        self.samplewise_init = True
        self.rotation = 20
        self.translation = 0.05
        self.deformation = None
        
        self.final = self.experiment_name + '_final.hdf5'
        self.best =  self.experiment_name + '_best.hdf5'
        self.callbacks = [CSVLogger( self.experiment_name + '.csv', append=True),
                          ModelCheckpoint(self.best, monitor='val_loss', verbose=2, save_best_only=True)]#,
                         #LearningRateScheduler(self.drop_decay, verbose=1)]
                    
