
import pandas as pd
import numpy as np
from numpy import array
import keras
from keras import backend as K
from keras.models import load_model
from keras.layers import Input, Dense, GRUCell, RNN, concatenate, TimeDistributed, Layer
import tensorflow as tf
from tensorflow.keras import initializers
tf.random.set_seed(69)
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def train_function(x):
    return np.sin(x)
    
def generate_train_sequences(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)+n_steps_out):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
        input_seq = array(X)
        output_seq = array(y)
    return input_seq, output_seq
    
# create sine data
xaxis = np.arange(-50*np.pi, 50*np.pi, 0.1)
train_seq = train_function(xaxis)

# set parameters
n_steps = 20
n_features = 1
idnn_epochs = 3
epochs = 8
batch_size = 32

# Initialize
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)

# input descriminator neural net (idnn).  This nn is used to build a memory of weights associated with which model (nn0 or nn1) does best when these weights are activated.
# Want it to be super fast and simple to start.
def create_model_idnn(layers):
    n_layers = len(layers)
    
    ## Encoder
    encoder_inputs = keras.layers.Input(shape=(None, 1))
    gru_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]

    encoder = keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    
    ## Decoder
    decoder_inputs = keras.layers.Input(shape=(None, 1))
    decoder_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]
    decoder_gru = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=encoder_states)
    [decoder_out, forward_h, forward_c] = decoder_gru(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense1 = Dense(10, activation='tanh', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs1 = decoder_dense1(decoder_out)
    decoder_dense2 = Dense(5, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs2 = decoder_dense2(decoder_out)
    
    merged = concatenate([decoder_outputs1, decoder_outputs2])
    
    decoder_dense = TimeDistributed(Dense(1, activation='sigmoid'))
    decoder_outputs = decoder_dense(merged)
    
    model_idnn = keras.models.Model([encoder_inputs,decoder_inputs], decoder_outputs)
    return model_idnn

# nn 0
def create_model_0(layers):
    n_layers = len(layers)
    
    ## Encoder
    encoder_inputs = keras.layers.Input(shape=(None, 1))
    gru_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]

    encoder = keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_outputs, state_h, state_h2, state_h3, state_c = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    
    ## Decoder
    decoder_inputs = keras.layers.Input(shape=(None, 1))
    decoder_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]
    decoder_gru = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=encoder_states)
    [decoder_out, forward_h, forward_h2, forward_h3, forward_c] = decoder_gru(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense1 = Dense(10, activation='tanh', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs1 = decoder_dense1(decoder_out)
    decoder_dense2 = Dense(5, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs2 = decoder_dense2(decoder_out)
    
    merged = concatenate([decoder_outputs1, decoder_outputs2])
    
    decoder_dense = TimeDistributed(Dense(1, activation='tanh'))
    decoder_outputs = decoder_dense(merged)
    
    model_0 = keras.models.Model([encoder_inputs,decoder_inputs], decoder_outputs)
    return model_0

# nn 1
def create_model_1(layers):
    n_layers = len(layers)
    
    ## Encoder
    encoder_inputs = keras.layers.Input(shape=(None, 1))
    gru_cells = [keras.layers.GRUCell(hidden_dim, activation='relu', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]

    encoder = keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_outputs, state_h, state_h2, state_h3, state_c = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    
    ## Decoder
    decoder_inputs = keras.layers.Input(shape=(None, 1))
    decoder_cells = [keras.layers.GRUCell(hidden_dim, activation='relu', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]
    decoder_gru = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=encoder_states)
    [decoder_out, forward_h, forward_h2, forward_h3, forward_c] = decoder_gru(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense1 = Dense(10, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs1 = decoder_dense1(decoder_out)
    decoder_dense2 = Dense(5, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs2 = decoder_dense2(decoder_out)
    
    merged = concatenate([decoder_outputs1, decoder_outputs2])
    
    decoder_dense = TimeDistributed(Dense(1, activation='relu'))
    decoder_outputs = decoder_dense(merged)
    
    model_1 = keras.models.Model([encoder_inputs,decoder_inputs], decoder_outputs)
    return model_1

 
neurons = 64

model_idnn = create_model_idnn([neurons,neurons])
model_0 = create_model_0([neurons,neurons,neurons,neurons])
model_1 = create_model_1([neurons,neurons,neurons,neurons])


batches = 1
input_seq_idnn, output_seq_idnn = generate_train_sequences(train_seq, n_steps, 1)
d = list() 
for i in range(len(output_seq_idnn)):
    decoder_target_data_idnn_initiate = np.random.randint(2, size=1)
    d.append(decoder_target_data_idnn_initiate)
    test = array(d)

def run_model_idnn(model,batches,epochs,batch_size):
    input_seq, output_seq = generate_train_sequences(train_seq, n_steps, 1)
    encoder_input_data = input_seq
    decoder_target_data = test 
    decoder_input_data = np.zeros(decoder_target_data.shape)
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.3,
        shuffle=False)
    total_loss_idnn.append(history.history['loss'])
    total_val_loss_idnn.append(history.history['val_loss'])
        
def run_model_0(model,batches,epochs,batch_size):

    for _ in range(batches):
        input_seq, output_seq = generate_train_sequences(train_seq, n_steps, 1)
        encoder_input_data = input_seq
        decoder_target_data = output_seq
        decoder_input_data = np.zeros(decoder_target_data.shape)
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=.3,
            shuffle=False)
        total_loss.append(history.history['loss'])
        total_val_loss.append(history.history['val_loss'])

def run_model_1(model,batches,epochs,batch_size):

    for _ in range(batches):
        input_seq, output_seq = generate_train_sequences(train_seq, n_steps, 1)
        encoder_input_data = input_seq
        decoder_target_data = output_seq
        decoder_input_data = np.zeros(decoder_target_data.shape)
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=.3,
            shuffle=False)
        total_loss_1.append(history.history['loss'])
        total_val_loss_1.append(history.history['val_loss'])

model_idnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')
model_0.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')

total_loss_idnn = []
total_val_loss_idnn = []

run_model_idnn(model_idnn,batches=batches, epochs=idnn_epochs,batch_size=batch_size)

# create test data and make predictions for idnn
idnn_xaxis = np.arange(0, 10*np.pi, 0.1)

def test_function(x):
    return np.sin(x)
    
seq_idnn = test_function(idnn_xaxis)

test_seq_idnn = seq_idnn[:n_steps]
results_idnn = []
for i in range(10): #len(test_xaxis) - n_steps):
    input_seq_test_idnn = test_seq_idnn[i:i+n_steps].reshape((1,n_steps,1))
    decoder_input_test_idnn = np.zeros((1,1,1))
    y_idnn = model_idnn.predict([input_seq_test_idnn, decoder_input_test_idnn])
    test_seq_idnn = np.append(test_seq_idnn, y_idnn)
        # scale data

sc=MinMaxScaler()
prediction_reshape = test_seq_idnn[-10:].reshape(-1,1)
DataScaler = sc.fit(prediction_reshape)
test_seq_idnn=DataScaler.transform(prediction_reshape)
print(test_seq_idnn)

# create test data and make predictions for either model_0 or model_1
test_xaxis = np.arange(0, 10*np.pi, 0.1)

def test_function(x):
    return np.sin(x)

seq = test_function(test_xaxis)

test_seq = seq[:n_steps]
results = []

fl_0 = []
fl_1 = []
for i in range(len(test_seq_idnn[-10:])):
    total_loss = []
    total_val_loss = []
    total_loss_1 = [] 
    total_val_loss_1 = []

    if test_seq_idnn[i] < .5:
        run_model_0(model_0,batches=batches, epochs=epochs,batch_size=batch_size)
        total_v_loss = [j for i in total_val_loss for j in i]
        final_loss_0 = total_v_loss[-1] # build new target data with this!!!!
        fl_0.append(final_loss_0)
        final_loss_array_0 = array(fl_0)
    else:
        run_model_1(model_1,batches=batches, epochs=epochs,batch_size=batch_size)
        total_v_loss_1 = [j for i in total_val_loss_1 for j in i]
        final_loss_1 = total_v_loss_1[-1] # build new target data with this!!!!
        fl_1.append(final_loss_1)
        final_loss_array_1 = array(fl_1)    

print(final_loss_array_0.mean())
print(final_loss_array_1.mean())

t0 = list()
t1 = list()
for i in range(len(output_seq_idnn)):
    tester_0 = [0]
    tester_1 = [1]
    t0.append(tester_0)
    t1.append(tester_1)
    zeros = array(t0)
    ones = array(t1)

# teach it with the best target data corresponding to the right model. 
if final_loss_array_0.mean() < final_loss_array_1.mean():
    # make target data of all 0's
    test = zeros
    run_model_idnn(model_idnn,batches=batches, epochs=epochs,batch_size=batch_size)
    for i in range(1):
        input_seq_test = test_seq[i:i+n_steps].reshape((1,n_steps,1))
        decoder_input_test = np.zeros((1,1,1))
        y = model_idnn.predict([input_seq_test, decoder_input_test])
else:
    # make target data of all 1's 
    test = ones
    run_model_idnn(model_idnn,batches=batches, epochs=epochs,batch_size=batch_size)
    for i in range(1):
        input_seq_test = test_seq[i:i+n_steps].reshape((1,n_steps,1))
        decoder_input_test = np.zeros((1,1,1))
        y = model_idnn.predict([input_seq_test, decoder_input_test])

if y < .5:
    for i in range(len(test_xaxis) - n_steps):
        input_seq_test = test_seq[i:i+n_steps].reshape((1,n_steps,1))
        decoder_input_test = np.zeros((1,1,1))
        y = model_0.predict([input_seq_test, decoder_input_test])
        test_seq = np.append(test_seq, y)
else:
    for i in range(len(test_xaxis) - n_steps):
        input_seq_test = test_seq[i:i+n_steps].reshape((1,n_steps,1))
        decoder_input_test = np.zeros((1,1,1))
        y = model_1.predict([input_seq_test, decoder_input_test])
        test_seq = np.append(test_seq, y)
        
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true = seq[n_steps:], y_pred = test_seq[n_steps:])

plt.plot(test_xaxis[n_steps:], test_seq[n_steps:], label="predictions")
plt.plot(test_xaxis, seq, label="ground truth")
plt.plot(test_xaxis[:n_steps], test_seq[:n_steps], label="initial sequence", color="red")
plt.title('GRU Approximation of Sine Function: MSE = ' + str(round(MSE,4)))
plt.legend(loc='upper left')
plt.ylim(-2, 2)
plt.show()
