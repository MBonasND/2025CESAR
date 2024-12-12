##################################################
##################################################
### Convolutional Neural Network (Autoencoder) ###
##################################################
##################################################

#import packages and functions
import numpy as np 
import pickle
import keras
from keras import optimizers
from keras.layers import Conv2DTranspose, Conv2D, LeakyReLU, LayerNormalization, TimeDistributed, Input, ConvLSTM2D, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import time
from functions import *

#request and initialize GPUS
gpus = tf.config.list_physical_devices('GPU')
strategy = tf.distribute.MirroredStrategy()
start_time = time.time()

#Specify simulation
index = 1
print()
print('Simulation:', index)
start_time = time.time()

#load smaller files --- Burger2D
pkl_file = open('burger2d.pkl', 'rb')
burger2d = pickle.load(pkl_file)
pkl_file.close()


simulation = index-1
burger2d_sim = np.transpose(burger2d[:,:,:,:,simulation], (0,2,3,1))
print()
print('Simulation Shape:', burger2d_sim.shape)


####################################################################################################### --- Build CAE


#get data for training and testing
test_size = 21
train_size = burger2d_sim.shape[0] - test_size


#scale data
burgermin, burgermax = burger2d_sim[:-test_size,...].min(), burger2d_sim[:-test_size,...].max()
burgernorm = (burger2d_sim - burgermin) / (burgermax - burgermin)
#burgernorm = np.concatenate([burgernorm[...,np.newaxis]], axis = -1)



print()
print('Train Size:', train_size)
print('Test Size:', test_size)
print('Input Train Shape:', burgernorm[:train_size].shape)
print('Output Train Shape:', burgernorm[:train_size].shape)


#Declare hyper-parameters
kernel_size = 3
#nodes_enc = [32,64,128]
nodes_enc = [16,32,64]
nodes_dec = list(reversed(nodes_enc))
activation = 'linear'
epochs = 500
batch_size = 2
dropout_rate = 0.9  # Specify the dropout rate


with strategy.scope():
    tf.random.set_seed(1997)

    #delcare inputs and first layer
    inputs = Input(shape=(burgernorm.shape[1:]), name='input')
    x = inputs

    #declare encoding layers
    for i, k in enumerate(nodes_enc):
        x = Conv2D(k, kernel_size=kernel_size, activation=activation, padding='same', name=f'Conv{i+1}', strides = 2)(x)
        x = LeakyReLU(name=f'ConvLeakyReLU{i+1}')(x)
        x = LayerNormalization(name=f'ConvNorm{i+1}')(x)
        x = Dropout(dropout_rate, name=f'ConvDropout{i+1}')(x)  # Add Dropout here

    #declare decoding layers
    for i, k in enumerate(nodes_dec):
        x = Conv2DTranspose(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'DeCo{len(nodes_enc)-i}')(x)
        x = LeakyReLU(name=f'DeCoLeakyReLU{i+1}')(x)
        x = LayerNormalization(name=f'DeCoNorm{len(nodes_enc)-i}')(x)
        x = Dropout(dropout_rate, name=f'DeCoDropout{i+1}')(x)  # Add Dropout here


    x = Conv2D(burgernorm.shape[-1], kernel_size=kernel_size, activation='sigmoid', padding='same', name='Output')(x)
    autoencoder = keras.Model(inputs, x, name='Autoencoder')
    #autoencoder.summary()

    #compile CNN
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.9), loss='mse')


###################################################################################################################### --- Train CAE


#decay the learning rate over based on the loss
reduce_lr = ReduceLROnPlateau(monitor='loss', 
                              factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                              patience=10,  # Number of epochs with no improvement after which learning rate will be reduced.
                              min_lr=1e-3, # Lower bound on the learning rate.
                              verbose=1)   # Verbosity mode. 0 = silent, 1 = update messages.


autoencoder.fit(burgernorm[:train_size], burgernorm[:train_size],
                batch_size=batch_size, epochs=epochs,
                verbose=0,
                callbacks=[reduce_lr]);


autoencoder.save(os.getcwd() + f'/CAE_Burger2D_{nodes_enc[0]}nh_Simulation{index-1}_MCDropout{dropout_rate}.keras')



################################################################################################################## --- Get Predictions


def predict_with_dropout(model, data, n_iter=100):
    """
    Function to make predictions with dropout enabled during inference.
    
    Parameters:
    model -- Trained Keras model with dropout layers.
    data -- Input data for prediction.
    n_iter -- Number of stochastic forward passes (default is 100).
    
    Returns:
    np.array of predictions from each iteration.
    """
    predictions = []
    for _ in range(n_iter):
        # Enable training mode (which also enables dropout)
        preds = model(data, training=True)
        predictions.append(preds)
    return tf.stack(predictions, axis=0)



#Get a collection of predictions:
n_iter = 100  # Number of stochastic forward passes
preds = predict_with_dropout(autoencoder, burgernorm[-test_size:], n_iter=n_iter)


print()
print('Predictions Shape:', preds.shape)

# Save the predictions from multiple runs
output = open(os.getcwd() + f'/CAE_Burger2D_predictions_Simulation{index-1}_MCDropout{dropout_rate}.pkl', 'wb')
pickle.dump(preds, output)
output.close()








