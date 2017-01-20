# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
import models
from keras import backend as K

def main(mode):
    # setup stuff to build model

    # This is it. use melgram, up to 6000 (SR is assumed to be 12000, see model.py),
    # do decibel scaling
    assert mode in ('feature', 'tagger')
    if mode == 'feature':
        last_layer = False
    else:
        last_layer = True

    args = Namespace(test=False, data_percent=100, model_name='', tf_type='melgram',
                     normalize='no', decibel=True, fmin=0.0, fmax=6000, 
                     n_mels=96, trainable_fb=False, trainable_kernel=False)

    model = models.build_convnet_model(args=args, last_layer=last_layer)
    model.load_weights('weights_{}.hdf5'.format(K._backend),
                       by_name=True)
    model.layers[1].summary()
    model.summary()
    # and use it!

if __name__ == '__main__':
    main('feature')
    main('tagger')
