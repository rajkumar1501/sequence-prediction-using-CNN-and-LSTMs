# -*- coding: utf-8 -*-
# predictive model used for predticion of miRNAs.
from __future__ import print_function
#import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D
from keras import regularizers
#from keras.utils import plot_model
import numpy as np
epochs = 75  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 19000  # Number of samples to train on.
num_encoder_tokens = 4
num_decoder_tokens = 6
max_encoder_seq_length = 29
max_decoder_seq_length = 28
  # Number of samples to train on.
# Path to the data txt file on disk.

input_token_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
target_token_index = {'\t': 4, '\n': 5, 'A': 0, 'C': 1, 'G': 2, 'U': 3}     
            
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
cnn = Conv1D(128,8, activation='relu')
cnn_output =cnn(encoder_inputs)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
encoder_dense_1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))
encoder_dense_output = encoder_dense_1(cnn_output)
encoder = LSTM(latent_dim, return_state=True,recurrent_dropout=0.4, dropout = 0.1)
encoder_outputs, state_h, state_c = encoder(encoder_dense_output)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
#cnn_decoder = Conv1D(128,8, activation='relu')
#cnn_decode_output =cnn_decoder(decoder_inputs)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
decoder_dense_1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))
decoder_dense_output = decoder_dense_1(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,recurrent_dropout=0.4, dropout = 0.1)
decoder_outputs, _, _ = decoder_lstm(decoder_dense_output,
                                     initial_state=encoder_states)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
#decoder_dense_1 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001))
#decoder_outputs = decoder_dense_1(decoder_outputs)
decoder_dense = Dense(num_decoder_tokens, activation='softmax',kernel_regularizer=regularizers.l2(0.001))
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
#plot_model(model,show_shapes=True, to_file = 'model_new.png')
#print(model.summary())           
# Run training

#model.load_weights('s2s_batch_50_energy_dim.h5')   
model.load_weights('s2s_batch_50_cnn_encoder_dense_dim.h5')
# Save model

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states
# Define sampling models
cnn_output = cnn(encoder_inputs)
encoder_dense_output = encoder_dense_1(cnn_output)
encoder_outputs, state_h, state_c = encoder(encoder_dense_output)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)


#cnn_decode_output =cnn_decoder(decoder_inputs)
#decoder_dense_output = decoder_dense_1(cnn_decode_output)
decoder_dense_output = decoder_dense_1(decoder_inputs)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_dense_output, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
#decoder_outputs = dropout_layer(decoder_outputs)
#decoder_outputs = decoder_dense_1(decoder_outputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
    return decoded_sentence

    
def predict_mirna(seq):    
    input_seq = seq
    
    New_encoder_input_data = np.zeros(
        ( 1,max_encoder_seq_length, num_encoder_tokens),
        dtype='float32') 
    for t,char in enumerate(input_seq):
            New_encoder_input_data[0,t, input_token_index[char]] = 1.
    
    
    return decode_sequence(New_encoder_input_data)
    


