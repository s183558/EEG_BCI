# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:49:44 2022

@author: Frederik
"""


##############################################################################
#                               Imports                                      #
##############################################################################

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal

# EEGNet-specific imports
from arl_EEGmodels.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


##############################################################################
#                             Functions                                      #
##############################################################################

def plot_eeg(EEG_data, vspace=0, channels = 6, color='k', samplerate = 250):
    '''
    Plot the EEG data, stacking the channels horizontally on top of each other.

    Parameters
    ----------
    EEG_data : array (channels x samples)
        The EEG data
    vspace : float (default 100)
        Amount of vertical space to put between the channels
    channels : int (default 6)
        Amount of  channels in the data set
    color : string (default 'k')
        Color to draw the EEG in
        
    Ripped from:
    https://notebook.community/joannekoong/neuroscience_tutorials/basic/1.%20Load%20EEG%20data%20and%20plot%20ERP
    '''
    
    # vspace * 0, vspace * 1,  ..., vspace * channels;
    bases = (vspace * np.arange(channels)).reshape(-1,1)
    
    # Center the data at 0
    EEG = EEG_data - EEG_data.mean(axis = 1, keepdims = 1)
    
    # To add the bases (a vector of length 7) to the EEG (a 2-D Matrix), we don't use
    # loops, but rely on a NumPy feature called broadcasting:
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    EEG += bases
    
    # Calculate a timeline in seconds, based on the EEG's sample rate
    time = np.arange(EEG.shape[1]) / samplerate
    
    # Plot EEG versus time
    plt.plot(time, EEG.T, color=color)

    # Add gridlines to the plot
    plt.grid()
    
    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    
    # The y-ticks are set to the locations of the electrodes. The international 10-20 system defines
    # default names for them.
    chan_names = ['Ch 0', 'Ch 1', 'Ch 2', 'Ch 3', 'Ch 4', 'Ch 5', 'Ch 6',
                  'Ch 7', 'Ch 8', 'Ch 9', 'Ch 10', 'Ch 11', 'Ch 12', 'Ch 13',
                  'Ch 14', 'Ch 15']
    plt.gca().yaxis.set_ticks(bases.flatten())
    plt.gca().yaxis.set_ticklabels(chan_names[:channels])
    
    # Put a nice title on top of the plot
    plt.title('EEG data')




##############################################################################
#                     Loading and filtering the data                         #
##############################################################################

if __name__ == '__main__':
    file_name   = 'OpenBCI-RAW-2022-01-05_15-41-56.txt'
    event_fname = 'event_annot.txt'
    
    raw_data    = pd.read_csv(file_name,skiprows = 4, parse_dates= True,
                              delimiter = ', ', engine = 'python')
    event_data  = np.loadtxt(event_fname, delimiter=',')
    
    data_cols   = raw_data.columns
    
    # Parameters
    sample_rate = 250 # Hz       The rate at which the data is sampled
    channels    = 6   #          Number of EEG channels
    
    
    #Remove the first row as it is just 0's and only take the EEG channels
    EXG_data  = raw_data[1:].filter(regex='EXG Channel')
    
    
   
    # Convert the data into a 2D np array
    X = EXG_data[1:].to_numpy()
    
    # Removing the samples which arent finished yet.
    freq                = 250 # Hz
    sample_size         = X.shape[0]
    unfinished_samples  = int(sample_size % freq)
    
    X                   = X[:-unfinished_samples,:]
    
    
    # Turning the np array into a 3D array, and transposing it so the channels 
    # become the rows, instead of the columns.
    X = X.reshape(int(X.shape[0]/freq), freq, 8).transpose(0,2,1)
    
    
    y = event_data.repeat(36)
    np.random.shuffle(y)
    
    # Gathering the attributes of the input data
    size = X.shape
    
    kernels = 1
    amount  = size[0]           # Amount of test
    chans   = size[1]           # Number of channels
    samples = size[2]           # The length of the measurement (the time for each)
    
    
    # Taking 50/25/25 percent of the data to train/validate/test
    slice_0_50    = slice(0, int(amount*0.5))
    slice_50_75   = slice(int(amount*0.5), int(amount*0.75))
    slice_75_100  = slice(int(amount*0.75), amount)
     
    
    X_train       = X[slice_0_50,]
    Y_train       = y[slice_0_50]
    X_validate    = X[slice_50_75,]
    Y_validate    = y[slice_50_75]
    X_test        = X[slice_75_100,]
    Y_test        = y[slice_75_100]
    
    
    ##############################################################################
    #                       Creating the EEGnet model                            #
    ##############################################################################
    
    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)
    
    # convert data to NHWC (trials, channels, samples, kernels) format. Data 
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
       
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout')
    
    
    # Create a plot Which displays the NN model 
    plot_model(model, to_file='model_plot.png', show_shapes=True,
               show_layer_names=True)
    
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    # count number of parameters in the model
    numParams    = model.count_params()    
    
    # set a valid path for your system to record model checkpoints
    model_filename  = 'best_model.h5'               #'test-Epoch-{epoch:02d}.h5'
    checkpoint_path = os.path.join('models/', model_filename)
    
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                   save_best_only=True)
    
    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################
    
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:1, 2:1, 3:1}
    
    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                            verbose = 2, validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight = class_weights)
    
    # load optimal weights
    model.load_weights(checkpoint_path)
    
    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################
    
    # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
    # model.load_weights(WEIGHTS_PATH)
    
    ###############################################################################
    # make prediction on test set.
    ###############################################################################
    
    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))
    
    
    ############################# PyRiemann Portion ##############################
    
    # code is taken from PyRiemann's ERP sample script, which is decoding in 
    # the tangent space with a logistic regression
    
    n_components = 2  # pick some components
    
    # set up sklearn pipeline
    clf = make_pipeline(XdawnCovariances(n_components),
                        TangentSpace(metric='riemann'),
                        LogisticRegression())
    
    preds_rg     = np.zeros(len(Y_test))
    
    # reshape back to (trials, channels, samples)
    X_train      = X_train.reshape(X_train.shape[0], chans, samples)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples)
    
    # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
    # labels need to be back in single-column format
    cov = Covariances('oas').fit_transform(X_train,Y_train.argmax(axis = -1))
    # clf.fit(X_train, Y_train.argmax(axis = -1))
    # preds_rg     = clf.predict(X_test)
    
    # # Printing the results
    # acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
    # print("Classification accuracy: %f " % (acc2))
    
    # plot the confusion matrices for both classifiers
    names        = ['Arm up', 'Arm down', 'Arm Right', 'Arm left']
    plt.figure(0)
    plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
    
    plt.figure(1)
    plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')













