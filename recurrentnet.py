# 2016-06-06 Updating for Keras 1.0 API
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape, Permute
from keras.layers.recurrent import GRU


def build_recurrentnet_model():
    ''' '''
    # -----some parameters
    num_hidden_rnn = 32
    num_rnn_layers = 2
    dim_labels = 50
    dropout_rnn_output = 0.3
    input_shape = (1, 96, 1366)
    # -----start a model and 'pre-process'
    model = Sequential()
    model.add(ZeroPadding2D(padding=(0, 37), input_shape=input_shape))
    model.add(BatchNormalization(axis=3))  # per frequency.
    # -----add convolutional layers
    model.add(get_convBNeluMPdrop((1, 96, 1440)))  # (None, num_feat, 1, 15)
    # -----add recurrent layers
    model.add(Permute((3, 1, 2)))  # (None, 15, num_feat, 1)
    model.add(Reshape((15, model.output_shape[2])))
    model.add(GRU(num_hidden_rnn, return_sequences=True))
    model.add(GRU(num_hidden_rnn, return_sequences=False))
    model.add(Dropout(dropout_rnn_output))
    # -----add output layer
    model.add(Dense(dim_labels, activation='sigmoid'))
    optimiser = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimiser)
    return model


def get_convBNeluMPdrop(input_shape):
    num_conv_layers = 4
    nums_feat_maps = [64, 128, 128, 128]
    pool_sizes = [(2, 2), (3, 3), (4, 4), (4, 4)]

    model = Sequential(name='ConvBNEluDr_4layer')
    input_shape_specified = False
    for conv_idx in xrange(num_conv_layers):
        if not input_shape_specified:
            model.add(Convolution2D(nums_feat_maps[conv_idx], 3, 3,
                                    input_shape=input_shape,
                                    border_mode='same',
                                    init='he_normal'))
            input_shape_specified = True
        else:
            model.add(Convolution2D(nums_feat_maps[conv_idx], 3, 3,
                                    border_mode='same',
                                    init='he_normal'))
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
        model.add(Dropout(0.5))
    return model
