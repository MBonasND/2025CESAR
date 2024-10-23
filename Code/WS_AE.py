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
from keras.layers import Conv2DTranspose, Conv2D, LeakyReLU, LayerNormalization, TimeDistributed, Input, ConvLSTM2D
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import time
from functions import *

#request and initialize GPUS
gpus = tf.config.list_physical_devices('GPU')
strategy = tf.distribute.MirroredStrategy()
start_time = time.time()

#get index value from CRC
index = int(os.getenv('SGE_TASK_ID'))
print('CRC array run:', index)
start_time = time.time()

#load smaller files --- magnitude
pkl_file = open('KAUST_magnitude_subset.pkl', 'rb')
magnitude = pickle.load(pkl_file)
pkl_file.close()




####################################################################################################### --- Build AE-ConvLSTM


#get data for training and testing
test_size = 24
train_size = magnitude.shape[0] - test_size


#scale data
windmin, windmax = magnitude[:-test_size,...].min(), magnitude[:-test_size,...].max()
windnorm = (magnitude - windmin) / (windmax - windmin)
windnorm = np.concatenate([windnorm[...,np.newaxis]], axis = -1)

print('Train Size:', train_size)
print('Test Size:', test_size)
print('Input Train Shape:', windnorm[:train_size].shape)
print('Output Train Shape:', windnorm[:train_size].shape)



#Declare hyper-parameters
kernel_size = 3
#nodes_enc = [32,64,128,256,2]
nodes_enc = [32,64,128]
nodes_dec = list(reversed(nodes_enc))
activation = 'linear'
epochs = 1000
batch_size = 10


with strategy.scope():
    tf.random.set_seed(1997)

    #delcare inputs and first layer
    inputs = Input(shape=(windnorm.shape[1:]), name='input')
    x = inputs

    #declare encoding layers
    for i, k in enumerate(nodes_enc):
        x = Conv2D(k, kernel_size=kernel_size, activation=activation, padding='same', name=f'Conv{i+1}', strides = 2)(x)
        x = LeakyReLU(name=f'ConvLeakyReLU{i+1}')(x)
        x = LayerNormalization(name=f'ConvNorm{i+1}')(x)

    #declare decoding layers
    for i, k in enumerate(nodes_dec):
        x = Conv2DTranspose(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'DeCo{len(nodes_enc)-i}')(x)
        x = LeakyReLU(name=f'DeCoLeakyReLU{i+1}')(x)
        x = LayerNormalization(name=f'DeCoNorm{len(nodes_enc)-i}')(x)


    x = Conv2D(1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='Output')(x)
    autoencoder = keras.Model(inputs, x, name='Autoencoder')
    autoencoder.summary()

    #compile CNN
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.9), loss='mse')




###################################################################################################################### --- Train AE


#decay the learning rate over based on the loss
reduce_lr = ReduceLROnPlateau(monitor='loss', 
                              factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                              patience=10,  # Number of epochs with no improvement after which learning rate will be reduced.
                              min_lr=1e-3, # Lower bound on the learning rate.
                              verbose=1)   # Verbosity mode. 0 = silent, 1 = update messages.


autoencoder.fit(windnorm[:train_size], windnorm[:train_size],
                batch_size=batch_size, epochs=epochs,
                verbose=0,
                callbacks=[reduce_lr]);


autoencoder.save(os.getcwd() + f'/models/deep_ae_{nodes_enc[0]}nh_subset_LeakyReLU_Run{index}.keras')



################################################################################################################## --- Get Predictions


#get autoencoder preds
results_autoencoder = autoencoder.predict(windnorm, verbose=0)
output = open(os.getcwd() + f'/forecasts/Deep_AE_predictions_LeakyReLU_Run{index}.pkl', 'wb')
pickle.dump(results_autoencoder, output)
output.close()

#slice testing points
preds_ae = ((windmax-windmin)*results_autoencoder[-test_size:,:,:,0] + windmin)
truth = ((windmax-windmin)*windnorm[-test_size:,:,:,0] + windmin)

#calculate mse for each location
print('Nodes:', nodes_enc)
print('RMSE of AE: {:.6f}'.format(np.sqrt(((truth-preds_ae)**2).mean())))




print()
train_time = time.time()
delta_train = train_time - start_time
delta_train_hrs = int(delta_train // (60*60))
delta_train_min = int( (delta_train % (60*60)) // 60) 
delta_train_sec = int(delta_train % 60)
print()
print(f'Training Time: {delta_train_hrs} hrs {delta_train_min} min {delta_train_sec} sec')
print()





