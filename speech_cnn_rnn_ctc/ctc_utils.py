import keras.backend as K
import tensorflow as tf

def decode(inputs, **kwargs):
    """ Decodes a sequence of probabilities choosing the path with highest
    probability of occur
    # Arguments
        is_greedy: if True (default) the greedy decoder will be used;
        otherwise beam search decoder will be used
        if is_greedy is False:
            see the documentation of tf.nn.ctc_beam_search_decoder for more
            options
    # Inputs
        A tuple (y_pred, seq_len) where:
            y_pred is a tensor (N, T, C) where N is the bath size, T is the
            maximum timestep and C is the number of classes (including the
            blank label)
            seq_len is a tensor (N,) that indicates the real number of
            timesteps of each sequence
    # Outputs
        A sparse tensor with the top path decoded sequence
    """
    is_greedy = kwargs.get('is_greedy', True)
    y_pred, input_length = inputs
    
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    input_length = tf.cast(input_length[:, 0], dtype='int32')

    if is_greedy:
        return tf.nn.ctc_greedy_decoder(y_pred, input_length)[0][0]
    else:
        beam_width = kwargs.get('beam_width', 100)
        top_paths = kwargs.get('top_paths', 1)
        return tf.nn.ctc_beam_search_decoder(y_pred, input_length, beam_width, top_paths)[0][0]

def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return (y_pred_shape[:1], None)

def ctc_lambda_func(args):
    """ CTC cost function
    """
    y_pred, labels, input_length = args

    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    input_length = tf.cast(input_length[:, 0], dtype='int32')
    
    return tf.nn.ctc_loss(inputs=y_pred, labels=labels, sequence_length=input_length)

def ctc_dummy_loss(y_true, y_pred):
    """ Little hack to make CTC working with Keras
    """
    return y_pred


def decoder_dummy_loss(y_true, y_pred):
    """ Little hack to make CTC working with Keras
    """
    return K.zeros((1,))