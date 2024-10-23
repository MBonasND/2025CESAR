#################################################
#################################################
### Functions to be used with the AE-ConvLSTM ###
#################################################
#################################################

#import packages
import numpy as np
import pickle
import os
import keras
import tensorflow as tf


############################### --- Generate data for input AE-ConvLSTM (sequence to sequence)

def gen_train_test_data_StS(data, look_back, look_fwd, test_size, batch_size):

    #extract training size --- uses all available data
    train_size = data.shape[0] - look_back - look_fwd  - test_size

    #normalize the data
    datamin, datamax = data[:-test_size,...].min(), data[:-test_size,...].max()
    datanorm = (data - datamin) / (datamax - datamin)
    datanorm = np.concatenate([datanorm[...,np.newaxis]], axis = -1)

    #compile data into input, output
    input_data = []
    output_data = []
    for t in range(look_back, datanorm.shape[0]-look_fwd):
        input_seq = datanorm[(t-look_back):t,...]
        if look_fwd > 1:
            output_seq = datanorm[t:(t+look_fwd),...]
        else:
            output_seq = datanorm[t,...]
        input_data.append(input_seq)
        output_data.append(output_seq)

    #convert to numpy arrays for slicing
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    #slice into training, valid, and testing data
    input_train = input_data[:train_size]
    output_train = output_data[:train_size]
    input_test = input_data[train_size:]
    output_test = output_data[train_size:]

    #prepare data for tf fitting
    data_train_tf = tf.data.Dataset.from_tensor_slices((input_train, output_train))
    data_train_tf = data_train_tf.batch(batch_size).prefetch(1)

    #prepare data for tf testing
    data_test_tf = tf.data.Dataset.from_tensor_slices((input_test, output_test))
    data_test_tf = data_test_tf.batch(batch_size).prefetch(1)

    #return all data
    return input_data, output_data, data_train_tf, data_test_tf



############################### --- Generate data for input AE-ConvLSTM (sequence to point)

def gen_train_test_data_StP(data, look_back, steps_ahead, test_size, batch_size):

    #extract training size --- uses all available data
    train_size = data.shape[0] - (look_back*steps_ahead) - test_size

    #normalize the data
    datamin, datamax = data[:-test_size,...].min(), data[:-test_size,...].max()
    datanorm = (data - datamin) / (datamax - datamin)
    datanorm = np.concatenate([datanorm[...,np.newaxis]], axis = -1)

    #compile data into input, output
    input_data = []
    output_data = []
    for t in range((look_back*steps_ahead), datanorm.shape[0]):
        wanted_range = range(t-(look_back*steps_ahead), t, steps_ahead)
        input_seq = datanorm[wanted_range,...]
        output_seq = datanorm[t,...]
        input_data.append(input_seq)
        output_data.append(output_seq)

    #convert to numpy arrays for slicing
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    #slice into training, valid, and testing data
    input_train = input_data[:train_size]
    output_train = output_data[:train_size]
    input_test = input_data[train_size:]
    output_test = output_data[train_size:]

    #prepare data for tf fitting
    data_train_tf = tf.data.Dataset.from_tensor_slices((input_train, output_train))
    data_train_tf = data_train_tf.batch(batch_size).prefetch(1)

    #prepare data for tf testing
    data_test_tf = tf.data.Dataset.from_tensor_slices((input_test, output_test))
    data_test_tf = data_test_tf.batch(batch_size).prefetch(1)

    #return all data
    return input_data, output_data, data_train_tf, data_test_tf


########################################## --- Extract Encoder and Decoder layers from AE

def get_AE_encoder_decoder(model):
    #extract encoder and decoder layers from AE
    encoder_layers = []
    decoder_layers = []

    #loop through CNN layers
    for ell in model.layers:
        if 'Conv' in ell.name:
            encoder_layers.append(ell)
        if 'DeCo' in ell.name:
            decoder_layers.append(ell)

    #export final layer to convert to original scale with outputs
    decoder_layers.append(model.layers[-1])

    return encoder_layers, decoder_layers





