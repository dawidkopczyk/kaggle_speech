import ctc_utils as ctc_utils

from keras.models import Model
from keras.activations import relu

from keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, 
                          Dense, Dropout, Activation, SimpleRNN, GaussianNoise,
                          Permute, GRU, TimeDistributed, Reshape, Bidirectional, Lambda)

def ctc_model(inputs, output, **kwargs):
    """ Given the input and output returns a model appending ctc_loss, the
    decoder, labels, and inputs_length
    # Arguments
        see core.ctc_utils.layer_utils.decode for more arguments
    """

    # Define placeholders
    labels = Input(name='labels', shape=[6], dtype='int32', sparse=True)
    inputs_length = Input(name='inputs_length', shape=[1], dtype='int32')
    
    # Define a decoder
    dec = Lambda(ctc_utils.decode, output_shape=ctc_utils.decode_output_shape,
                 arguments={'is_greedy': True}, name='decoder')
    y_pred = dec([output, inputs_length])

    # Define loss as a layer
    ctc = Lambda(ctc_utils.ctc_lambda_func, output_shape=(1,), name="ctc")
    loss = ctc([output, labels, inputs_length])

    return Model(inputs=[inputs, labels, inputs_length], outputs=[loss, y_pred])


def deep_ctc(features_shape, num_classes, act='relu'):

    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv1', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)
    
    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv1')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv1')(o)
    o = MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Block 4
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block4_conv1')(o)
    o = MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block4_pool')(o)
    o = BatchNormalization(name='block4_norm')(o)

    # Block 5
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block5_conv1')(o)
    o = MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block5_pool')(o)
    o = BatchNormalization(name='block5_norm')(o)
    
    # Block RNN
    o = Permute((2, 1, 3), name='rnn_permute')(o)
    o = Reshape((98, 4*32), name='rnn_reshape')(o)
    o = Bidirectional(GRU(units=256, return_sequences=True), name='rnn_bigru1')(o)
    o = TimeDistributed(BatchNormalization(), name='rnn_norm1')(o)
    o = Bidirectional(GRU(units=256, return_sequences=True), name='rnn_bigru2')(o)
    o = TimeDistributed(BatchNormalization(), name='rnn_norm2')(o)
#    o = TimeDistributed(Dropout(0.5), name='rnn_drop')(o)
    
    # Predictions
    o = TimeDistributed(Dense(num_classes), name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return ctc_model(x, o)

def deep_speech(features_shape, num_classes, num_hiddens=512, dropout=0.1,
                max_value=20):

    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    o = Permute((2, 1))(o)
    
    def clipped_relu(x):
        return relu(x, max_value=max_value)

    # First layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = TimeDistributed(Activation(relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Second layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = TimeDistributed(Activation(relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Third layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = TimeDistributed(Activation(relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fourth layer
    o = Bidirectional(GRU(units=num_hiddens, return_sequences=True))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fifth layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = TimeDistributed(Activation(relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Output layer
    o = TimeDistributed(Dense(num_classes))(o)

    Model(inputs=x, outputs=o).summary()
    
    return ctc_model(x, o)