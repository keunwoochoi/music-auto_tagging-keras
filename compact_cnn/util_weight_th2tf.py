''' util: convert theano weight file to tensorflow weight file
'''
from keras import backend as K
from keras.utils.np_utils import convert_kernel
import tensorflow as tf
from argparse import Namespace
import models


def main():

    if K._backend == 'theano':
        raise RuntimeError('Set backend as TF')
    args = Namespace(test=False, data_percent=100, model_name='', tf_type='melgram',
                     normalize='no', decibel=True, fmin=0.0, fmax=6000, 
                     n_mels=96, trainable_fb=False, trainable_kernel=False)

    model = models.build_convnet_model(args=args, last_layer=False)
    model.load_weights('weights_theano.hdf5',
                       by_name=True)

    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', \
         'conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
        if layer.__class__.__name__ in ['ConvBNEluDr', 'Sequential']:
            for sub_layer in layer.layers:
                if sub_layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
                    original_w = K.get_value(sub_layer.W)
                    converted_w = convert_kernel(original_w)
                    ops.append(tf.assign(sub_layer.W, converted_w).op)

    if ops == []:
        raise RuntimeError('no operation')
    K.get_session().run(ops)

    model.save_weights('weights_tensorflow.hdf5')

if __name__ == '__main__':

    main()
