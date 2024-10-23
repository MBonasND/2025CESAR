#######################################
#######################################
### AE-ConvLSTM (sequence to point) ###
#######################################
####################################### 

#import packages and functions
import numpy as np 
import pickle
import keras
from keras import optimizers
from keras.layers import Conv2DTranspose, Conv2D, LeakyReLU, LayerNormalization, TimeDistributed, Input, ConvLSTM2D
from keras.layers import BatchNormalization, Dense, Flatten, Reshape, Permute, Input, Lambda, RepeatVector, Dropout, RNN, Softmax
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import time
from functions import *




tf.config.run_functions_eagerly(False)

########################################################## --- Build ESN Class for tf

class EchoStateRNNCell(keras.layers.Layer):
    def __init__(self, units, decay=0.1, alpha=0.5, rho=1.0, sw=1.0, seed=None,
                 epsilon=None, sparseness=0.0,  activation=None, optimize=False,
                 optimize_vars=None, *args, **kwargs):

        self.seed = seed
        self.units = units
        self.state_size = units
        self.sparseness = sparseness
        self.decay_ = decay
        self.alpha_ = alpha
        self.rho_ = rho
        self.sw_ = sw
        self.epsilon = epsilon
        self._activation = tf.tanh if activation is None else activation
        self.optimize = optimize
        self.optimize_vars = optimize_vars

        super(EchoStateRNNCell, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.optimize_table = {"alpha": False, "rho": False, "decay": False, "sw": False}

        if self.optimize == True:
            for var in ["alpha", "rho", "decay", "sw"]:
                if var in self.optimize_vars:
                    self.optimize_table[var] = True
                else:
                    self.optimize_table[var] = False

        self.decay = tf.Variable(self.decay_, name="decay", dtype=tf.float32, trainable=self.optimize_table["decay"])
        self.alpha = tf.Variable(self.alpha_, name="alpha", dtype=tf.float32, trainable=self.optimize_table["alpha"])
        self.rho = tf.Variable(self.rho_, name="rho", dtype=tf.float32, trainable=self.optimize_table["rho"])
        self.sw = tf.Variable(self.sw_, name="sw", dtype=tf.float32, trainable=self.optimize_table["sw"])
        self.alpha_store = tf.Variable(self.alpha_, name="alpha_store", dtype=tf.float32, trainable=False) 
        self.echo_ratio = tf.Variable(1, name="echo_ratio", dtype=tf.float32, trainable=False) 

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer=keras.initializers.RandomUniform(-1, 1, seed=self.seed),
                                      name="kernel", trainable=False)

        self.recurrent_kernel_init = self.add_weight(shape=(self.units, self.units), initializer=keras.initializers.RandomNormal(seed=self.seed),
                                                    name="recurrent_kernel", trainable=False)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer=tf.zeros_initializer(), name="recurrent_kernel", trainable=False)

        self.recurrent_kernel_init.assign(self.setSparseness(self.recurrent_kernel_init))
        self.recurrent_kernel.assign(self.setAlpha(self.recurrent_kernel_init))
        self.echo_ratio.assign(self.echoStateRatio(self.recurrent_kernel))
        self.rho.assign(self.findEchoStateRho(self.recurrent_kernel * self.echo_ratio))

        self.built = True

    def setAlpha(self, W):
        return 0.5 * (self.alpha * (W + tf.transpose(W)) + (1 - self.alpha) * (W - tf.transpose(W)))

    def setSparseness(self, W):
        mask = tf.cast(tf.random.uniform(W.shape, seed=self.seed) < (1 - self.sparseness), dtype=W.dtype)
        return W * mask

    def echoStateRatio(self, W):
        eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
        return tf.reduce_max(tf.abs(eigvals))

    def findEchoStateRho(self, W):
        target = 1.0
        eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
        x = tf.math.real(eigvals)
        y = tf.math.imag(eigvals)

        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        sol = (tf.sqrt(b**2 - 4*a*c) - b) / (2*a)
        rho = tf.reduce_min(sol)
        return rho

    def clip_variables(self):
        self.decay.assign(tf.clip_by_value(self.decay, 0.00000001, 0.25))
        self.alpha.assign(tf.clip_by_value(self.alpha, 0.000001, 0.9999999))
        self.rho.assign(tf.clip_by_value(self.rho, 0.5, 1.0e100))
        self.sw.assign(tf.clip_by_value(self.sw, 0.5, 1.0e100))

    def call(self, inputs, states): 
        rkernel = self.setAlpha(self.recurrent_kernel_init)
        condition = tf.not_equal(self.alpha, self.alpha_store)
        def true_fn():
            self.clip_variables()
            self.echo_ratio.assign(self.echoStateRatio(rkernel))
            self.rho.assign(self.findEchoStateRho(rkernel * self.echo_ratio)) 
            self.alpha_store.assign(self.alpha)
            return rkernel
        def false_fn():
            return rkernel
        rkernel = tf.cond(condition, true_fn, false_fn)

        ratio = self.rho * self.echo_ratio * (1 - self.epsilon)
        prev_output = states[0]
        output = prev_output + self.decay * (tf.matmul(inputs, self.kernel * self.sw) + tf.matmul(self._activation(prev_output), rkernel * ratio) - prev_output)
        return self._activation(output), [output]

    def get_config(self):
        config = super(EchoStateRNNCell, self).get_config()
        config.update({
            'units': self.units,
            'decay': self.decay_,
            'alpha': self.alpha_,
            'rho': self.rho_,
            'sw': self.sw_,
            'seed': self.seed,
            'epsilon': self.epsilon,
            'sparseness': self.sparseness,
            'activation': self._activation,
            'optimize': self.optimize,
            'optimize_vars': self.optimize_vars
        })
        return config





####################################################### --- Preliminaries


#request and initialize GPUS
gpus = tf.config.list_physical_devices('GPU')
strategy = tf.distribute.MirroredStrategy()
start_time = time.time()

#get index value from CRC
index = int(os.getenv('SGE_TASK_ID'))
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



####################################################################################################### --- Build AE-ConvLSTM


#get data for training and testing
look_back = 4
test_size = 21
steps_ahead = 1
batch_size = 2
train_size = burger2d_sim.shape[0] - (look_back*steps_ahead) - test_size

#scale data
burgermin, burgermax = burger2d_sim[:-test_size,...].min(), burger2d_sim[:-test_size,...].max()
burgernorm = (burger2d_sim - burgermin) / (burgermax - burgermin)
input_data, output_data, data_train_tf, data_test_tf = gen_train_test_data_StP(burgernorm, look_back, steps_ahead, test_size, batch_size)

print('Train Size:', train_size)
print('Input Data Shape:', input_data.shape)
print('Output Data Shape:', output_data.shape)
print('Input Train Shape:', input_data[:train_size].shape)
print('Output Train Shape:', output_data[:train_size].shape)


#Declare hyper-parameters
kernel_size = 3
nodes_enc = [16,32,64]
nodes_dec = list(reversed(nodes_enc))
activation = 'linear'
epochs = 500


#load pretrained model - Autoencoder
with strategy.scope():
    autoencoder = tf.keras.models.load_model(os.getcwd() + f'/models/AE_Burger2D_{nodes_enc[0]}nh_Simulation{index-1}.keras')


#extract encoder and decoder layers from AE
encoder_layers = []
decoder_layers = []

#loop through CNN layers
for ell in autoencoder.layers:
    if 'Conv' in ell.name:
        encoder_layers.append(ell)
    if 'DeCo' in ell.name:
        decoder_layers.append(ell)

#export final layer to convert to original scale with outputs
decoder_layers.append(autoencoder.layers[-1])


#Build in ConvLSTM component
with strategy.scope():
    tf.random.set_seed(1997)
    
    #declare input
    inputs = Input(shape=(input_data.shape[1:-1]), name='input')
    
    #reformulate encoder layers as time distributed
    x = TimeDistributed(encoder_layers[0], name=encoder_layers[0].name)(inputs)
    for l in encoder_layers[1:]:
        x = TimeDistributed(l, name=l.name)(x)
    
    #reshape encoded features
    shape = list(x.shape[1:])
    flattened_dim = int(tf.reduce_prod(x.shape[2:]))  # Flatten dimensions from 2 onwards
    x = Reshape((x.shape[1], flattened_dim), name='ReshapeESNPre')(x)

    #Initialize ESN layers and get back into correct shape
    cell = EchoStateRNNCell(units=shape[-1], activation=lambda x: tf.math.tanh(x), decay=0.1,  epsilon=1e-20, alpha=0.5,
                            optimize=True, optimize_vars=["rho", "decay", "alpha", "sw"])
    x = RNN(cell, return_sequences=False, name="RNNESN")(x)
    x = Dense(np.prod(shape[1:]), activation=activation, name='DenseESN')(x)
    x = Reshape((shape[1:]), name='ReshapeESNPost')(x)

    
    
    #reformualte decoder layers here
    for l in decoder_layers:
        x = l(x)
    
    #finalize model and declare original AE parameters as non-trainable
    conv_esn = keras.Model(inputs, x, name='AE_ESN')
    for i, l in enumerate(conv_esn.layers):
        if 'ESN' not in l.name:
            l.trainable = False #we've already trained the autoencoder, we just need the ESN parameters

    #print model summary
    # conv_esn.summary()

    #compile the model
    conv_esn.compile(optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.9), loss='mse')



################################################################################################################### --- Train CNN w/ ConvLSTM




#decay the learning rate over based on the loss
reduce_lr = ReduceLROnPlateau(monitor='loss', 
                            factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                            patience=10,  # Number of epochs with no improvement after which learning rate will be reduced.
                            min_lr=1e-3, # Lower bound on the learning rate.
                            verbose=1)   # Verbosity mode. 0 = silent, 1 = update messages.



conv_esn.fit(data_train_tf,
              epochs=epochs,
              verbose=0,
              callbacks=[reduce_lr]);



conv_esn.save(os.getcwd() + f'/models/conv_ESN_{nodes_enc[0]}nh_Simulation{index-1}_StP.keras')


################################################################################################################## --- Get Predictions


#get conv_lstm preds
#results_conv_lstm = conv_lstm.predict(input_data, verbose=0)
#output = open(os.getcwd() + f'/models/Conv_LSTM_predictions_LeakyReLU_Run{index}_Tau{steps_ahead}_StP.pkl', 'wb')
#pickle.dump(results_conv_lstm, output)
#output.close()


# #slice testing points
#print()
#print()
#print(f'Standard forecasting {steps_ahead}-steps ahead...')
#print('Lags Used:', look_back)
#print('Steps Ahead:', steps_ahead)
#preds = ((windmax-windmin)*results_conv_lstm[-test_size:,:,:,0] + windmin)
#truth = ((windmax-windmin)*windnorm[-test_size:,:,:,0] + windmin)

#calculate mse for each location
#print('RMSE of ConvLSTM (Standard): {:.6f}'.format(np.sqrt(((truth-preds)**2).mean())))
#print()


################################################################################################################# --- Get Predictions (recursively)


#predict next look_fwd after end of training data
predictions = conv_esn.predict(input_data[:,:,:,:,:,0], verbose=0)
last_input = input_data[train_size:(train_size+steps_ahead),-(look_back-1):,:,:,:,0]


#recursively forecast remaining test points
results_conv_esn =  predictions[train_size:(train_size+steps_ahead),:,:,:]
new_input = np.concatenate([last_input, np.expand_dims(results_conv_esn, axis = 1)], axis = 1)
loops = int((test_size / steps_ahead) - 1)
for j in range(loops):
    predictions = conv_esn.predict(new_input, verbose = 0)
    results_conv_esn = np.concatenate([results_conv_esn, predictions], axis = 0)
    new_input = np.concatenate([new_input[:,-(look_back-1):,:,:,:], np.expand_dims(predictions, axis = 1)], axis = 1)   


#save predictions output
output = open(os.getcwd() + f'/forecasts/Conv_ESN_predictions_Simulation{index-1}_StP_Recursive.pkl', 'wb')
pickle.dump(results_conv_esn, output)
output.close()


#slice testing points
print()
print(f'Recursive forecasting {steps_ahead}-steps ahead for {test_size} future points...')
print('Lags Used:', look_back)
#print('Steps Ahead:', steps_ahead)
preds = ((burgermax-burgermin)*results_conv_esn[:,:,:,0] + burgermin)
truth = ((burgermax-burgermin)*burgernorm[-test_size:,:,:,0] + burgermin)

#calculate mse for each location
scalar = 1000
print()
print('Median MSE of ConvESN (Recursive): {:.6f}'.format(np.median(((truth-preds)**2).mean(axis = 0))*scalar))