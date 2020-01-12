# -*- coding: utf-8 -*-
"""
CODE TO IMPLEMENT A VAD (VOICE ACTIVITY DETECTOR) USING 
BINARIZED NEURAL NETWORK 

Abstract:
    (1) The following code implements a Binary Net quantized to {0,1} - binary_sigmoid 
    (or) {-1,1} - binarh_tanh. binarize() function in binary_ops also to be modified 
    accordingly.
    Data to be trained is loaded using data() from a '.mat' file
    containing features and corresponding labels.
    
    (2) A sequential model is implemented and trained using 'mse' cost (mean squared error) and 
    ADAM optimiser rule, learning rate - 0.001. 
    
    (3) Hyperopt library has been used to find the model that has the highest validation 
    accuracy amongst the training epochs.
    
    (4) Input - 16 neurons analog values
        Output - 1 neuron digital - 0/1
    
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform
import tensorflow as tf

import keras.backend as K
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

from scipy.io import loadmat

# Fully connected layer with Binary Weights 
from binary_layers import BinaryDense
# Quantizing to {-1,1}
from binary_ops import binary_tanh
# Quantizing to {0,1}
from binary_ops import binary_sigmoid 

from sklearn.metrics import confusion_matrix

# Code to find the True Positive rate & True Negative rate    

def metric(y_train,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_train.reshape(len(y_train),
                                                         1),y_pred).ravel()
    
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    return tpr,tnr,(tp+tn)/(tp+tn+fp+fn)

def data():
    snr='15'
    noise_context='babble'
    
    data_train = loadmat('F:/ProgramFiles/MATLAB/VADSohn/train_speech_'+snr+'_'+noise_context+'.mat')['data1']
    data_test = loadmat('F:/ProgramFiles/MATLAB/VADSohn/train_speech_'+snr+'_'+noise_context+'.mat')['data2']
    
    clean = loadmat('F:/ProgramFiles/MATLAB/VADSohn/train_speech_'+snr+'_'+noise_context+'.mat')['audio_clean']
   
    # X - Contains 16 analog features
    # Y - Contains label - Speech (1)/ Non-speech(0)
    
    X_train = data_train[:,:-1]
    y_train = data_train[:,-1]
    
    X_test = data_test[:,:-1]
    y_test = data_test[:,-1]

    Y_train = y_train 
    Y_test = y_test 

    return X_train, Y_train, X_test, Y_test,y_train,y_test,clean

def create_model(X_train, Y_train, X_test, Y_test):
    
    # Hyperparameters
    
    H = 'Glorot'
    kernel_lr_multiplier = 'Glorot'
    use_bias = False
    epsilon = 1e-3
    momentum = 0.9
    epochs = 10
    batch_size = {{choice([512])}}
    
    # Number of units per layer 
    N = 64       
    
    # Building the model
    
    model = Sequential()
    # Input layer
    model.add(BatchNormalization(input_shape=(16,), momentum=momentum, epsilon=epsilon))
    model.add(Activation(binary_sigmoid))
    
    # Hidden Layer 1
    model.add(BinaryDense(N, W_regularizer=l2(0.0), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
    model.add(BatchNormalization())
    model.add(Activation(binary_sigmoid))
    # Hidden Layer 2
    model.add(BinaryDense(N, W_regularizer=l2(0.0), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
    model.add(BatchNormalization())
    model.add(Activation(binary_sigmoid))
    # Hidden Layer 3
    model.add(BinaryDense(N, W_regularizer=l2(0.0), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
    model.add(BatchNormalization())
    model.add(Activation(binary_sigmoid))
    # Hidden Layer 4
    model.add(BinaryDense(N, W_regularizer=l2(0.0), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
    model.add(BatchNormalization())
    model.add(Activation(binary_sigmoid))
    # Output layer 
    model.add(BinaryDense(1, W_regularizer=l2(0.0), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias ))    
    model.add(BatchNormalization())
    model.add(Activation(binary_sigmoid))
    
    # Optimiser & cost functions
    opt=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse',optimizer=opt,metrics=['binary_accuracy'])
    
    model.summary()
     
    result = model.fit(X_train, Y_train,batch_size=batch_size,epochs= epochs,
                       verbose=2,
                       validation_data=(X_test, Y_test))
    
    # Plotting training curve
    
    acc = result.history['binary_accuracy']
    val_acc = result.history['val_binary_accuracy']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy & Loss History')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.show()
    
    validation_acc = np.amax(result.history['val_binary_accuracy']) 
    tr_loss = np.amax(result.history['loss'])
    print('Best validation acc of epoch:', validation_acc)
    
    return {'loss': tr_loss, 'status': STATUS_OK, 'model': model}
      
if __name__ == '__main__':
    
    TP=[]
    TN=[]
    ACC=[]
    
    for i in range(1):
        
        # Calling hyperopt library to get the maximum validation accuracy parameters
        best_run, best_model = optim.minimize(model=create_model,
                                                  data=data,
                                                  algo=tpe.suggest,
                                                  max_evals= 1,
                                                  trials=Trials())
                   
        X_train, Y_train, X_test, Y_test, y_train, y_test,audio_clean=data() 
        
        # Inference on Test dataset and calculating Hit rates

        y_pred = best_model.predict_classes(X_test)
        tpr,tnr,acc=metric(y_test,y_pred)
        print('Test accuracy:',tpr,tnr,acc)
        TP.append(tpr)
        TN.append(tnr)
        ACC.append(acc)
        print(tpr,tnr,acc)
    print(TP)
    print(TN)
    print(ACC)
    
    # Plotting audio and classification sequence
    
    snr='15'
    noise_context='babble'
    audio_noise = loadmat('F:/ProgramFiles/MATLAB/VADSohn/train_speech_'+snr+'_'+noise_context+'.mat')['audio_noisy']

    plt.plot(audio_noise[:10000])
    plt.plot(y_pred[:10000])
    plt.show()
    
    plt.plot(audio_clean[:10000])
    plt.plot(y_test[:10000])
    plt.show()
    
    