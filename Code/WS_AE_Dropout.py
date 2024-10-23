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

#get index value from CRC
index = int(os.getenv('SGE_TASK_ID'))
print()
print('CRC array run:', index)
start_time = time.time()

#load smaller files --- magnitude
pkl_file = open('KAUST_magnitude_subset.pkl', 'rb')
magnitude = pickle.load(pkl_file)
pkl_file.close()


#load and  combine all of the data
#splits = 10
#arrays_list = []
#for dat in range(splits):
#    filename = os.getcwd() + f'/full_data/KAUST_magnitude_subset_5min_Part{dat}.pkl'
#    # Load the array from the pickle file
#    with open(filename, 'rb') as f:
#        array = pickle.load(f)
#        arrays_list.append(array)

#magnitude = np.concatenate(arrays_list, axis=0)
print()
print('Full Data Shape:', magnitude.shape)


####################################################################################################### --- Build AE-ConvLSTM


#get data for training and testing
test_size = 24 #CHANGE IF I DONT USE THE FULL DATA
train_size = magnitude.shape[0] - test_size


#scale data
windmin, windmax = magnitude[:-test_size,...].min(), magnitude[:-test_size,...].max()
windnorm = (magnitude - windmin) / (windmax - windmin)
windnorm = np.concatenate([windnorm[...,np.newaxis]], axis = -1)

print()
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
dropout_rate = 0.2  # Specify the dropout rate


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
        x = Dropout(dropout_rate, name=f'ConvDropout{i+1}')(x)  # Add Dropout here

    #declare decoding layers
    for i, k in enumerate(nodes_dec):
        x = Conv2DTranspose(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'DeCo{len(nodes_enc)-i}')(x)
        x = LeakyReLU(name=f'DeCoLeakyReLU{i+1}')(x)
        x = LayerNormalization(name=f'DeCoNorm{len(nodes_enc)-i}')(x)
        x = Dropout(dropout_rate, name=f'DeCoDropout{i+1}')(x)  # Add Dropout here


    x = Conv2D(1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='Output')(x)
    autoencoder = keras.Model(inputs, x, name='Autoencoder')
    #autoencoder.summary()

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


autoencoder.save(os.getcwd() + f'/dropout/AE_{nodes_enc[0]}nh_MCDropout_Run{index}.keras')



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
preds = predict_with_dropout(autoencoder, windnorm[-test_size:], n_iter=n_iter)


print()
print('Predictions Shape:', preds.shape)

# Save the predictions from multiple runs
output = open(os.getcwd() + f'/dropout/AE_predictions_MCDropout_Run{index}.pkl', 'wb')
pickle.dump(preds, output)
output.close()


# Calculate the mean and quantiles for the prediction intervals
mean_preds = np.mean(preds, axis=0)
lower_bound = np.percentile(preds, 2.5, axis=0)  # 2.5th percentile for the lower bound (95% PI)
upper_bound = np.percentile(preds, 97.5, axis=0)  # 97.5th percentile for the upper bound (95% PI)


print()
print('Mean Shape:', mean_preds.shape)
print('Lower Shape:', lower_bound.shape)
print('Upper Shape:', upper_bound.shape)

# Save the results
output = open(os.getcwd() + f'/dropout/AE_prediction_intervals_MCDropout_Run{index}.pkl', 'wb')
pickle.dump({'mean_preds': mean_preds, 'lower_bound': lower_bound, 'upper_bound': upper_bound}, output)
output.close()







