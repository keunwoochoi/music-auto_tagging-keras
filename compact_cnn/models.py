# 2016-06-06 Updating for Keras 1.0 API
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Layer, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

SR = 12000


def build_convnet_model(args, last_layer=True, sr=None, compile=True):
    ''' '''
    tf = args.tf_type
    normalize = args.normalize
    if normalize in ('no', 'False'):
        normalize = None
    decibel = args.decibel
    model = raw_vgg(args, tf=tf, normalize=normalize, decibel=decibel,
                    last_layer=last_layer, sr=sr)
    if compile:
        model.compile(optimizer=keras.optimizers.Adam(lr=5e-3),
                      loss='binary_crossentropy')
    return model


def raw_vgg(args, input_length=12000 * 29, tf='melgram', normalize=None,
            decibel=False, last_layer=True, sr=None):
    ''' when length = 12000*29 and 512/256 dft/hop, 
    melgram size: (n_mels, 1360)
    '''
    assert tf in ('stft', 'melgram')
    assert normalize in (None, False, 'no', 0, 0.0, 'batch', 'data_sample', 'time', 'freq', 'channel')
    assert isinstance(decibel, bool)

    if sr is None:
        sr = SR  # assumes 12000

    conv_until = args.conv_until
    trainable_kernel = args.trainable_kernel
    model = Sequential()
    # decode args
    fmin = args.fmin
    fmax = args.fmax
    if fmax == 0.0:
        fmax = sr / 2
    n_mels = args.n_mels
    trainable_fb = args.trainable_fb
    model.add(Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, input_length),
                             trainable_kernel=trainable_kernel,
                             trainable_fb=trainable_fb,
                             return_decibel_melgram=decibel,
                             sr=sr, n_mels=n_mels,
                             fmin=fmin, fmax=fmax,
                             name='melgram'))

    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]

    if normalize in ('batch', 'data_sample', 'time', 'freq', 'channel'):
        model.add(Normalization2D(normalize))
    model.add(get_convBNeluMPdrop(5, [32, 32, 32, 32, 32],
                                  [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                                  poolings, model.output_shape[1:], conv_until=conv_until))
    if conv_until != 4:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())

    if last_layer:
        model.add(Dense(50, activation='sigmoid'))
    return model


def get_convBNeluMPdrop(num_conv_layers, nums_feat_maps,
                        conv_sizes, pool_sizes, input_shape, conv_until=None):
    # [Convolutional Layers]
    model = Sequential(name='ConvBNEluDr')
    input_shape_specified = False

    if conv_until is None:
        conv_until = num_conv_layers  # end-inclusive.

    for conv_idx in xrange(num_conv_layers):
        # add conv layer
        if not input_shape_specified:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                    conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                    input_shape=input_shape,
                                    border_mode='same',
                                    init='he_normal'))
            input_shape_specified = True
        else:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                    conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                    border_mode='same',
                                    init='he_normal'))
        # add BN, Activation, pooling
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(keras.layers.advanced_activations.ELU(alpha=1.0))  # TODO: select activation

        model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
        if conv_idx == conv_until:
            break

    return model
