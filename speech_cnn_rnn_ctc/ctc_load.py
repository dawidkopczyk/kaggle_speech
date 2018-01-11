import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf

import ctc_utils

def to_dense(x):
    if K.is_sparse(x):
        return tf.sparse_tensor_to_dense(x, default_value=24)
    return x

def to_dense_output_shape(input_shape):
    return input_shape

def append_dummy(x):
    x_shape = tf.shape(x)
    x_append = tf.tile(tf.constant([[24]], dtype='int64'), [x_shape[0], 98-x_shape[1]])
    x = tf.concat([x, x_append], 1)
    return x

def append_dummy_output_shape(input_shape):
    return input_shape

def load_weights(model, model_fname, mode='train', **kwargs):
    if mode not in ('train', 'predict'):
        raise ValueError('Mode must be one of (train, predict)')

    model.load_weights(model_fname)

    if mode == 'predict':
        # Define the new decoder and the to_dense layer
        dec = Lambda(ctc_utils.decode,
                     output_shape=ctc_utils.decode_output_shape,
                     arguments={'is_greedy': kwargs.get('is_greedy', False),
                                'beam_width': kwargs.get('beam_width', 400)},
                     name='beam_search')


        y_pred = model.get_layer('decoder').input[0]

        input_ = model.get_layer('inputs').input
        inputs_length = model.get_layer('inputs_length').input

        to_dense_layer = Lambda(
            to_dense,
            output_shape=to_dense_output_shape,
            name="to_dense")
        
        append_dummy_layer = Lambda(
            append_dummy,
            output_shape=append_dummy_output_shape,
            name="append_dummy")

        y_pred = dec([y_pred, inputs_length])

        y_pred = to_dense_layer(y_pred)
        y_pred = append_dummy_layer(y_pred)
        
        model = Model(inputs=[input_, inputs_length],
                      outputs=[y_pred])

    return model
    