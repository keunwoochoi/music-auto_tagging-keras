import time
import numpy as np
from keras import backend as K
from music_tagger_cnn import MusicTaggerCNN
from music_tagger_crnn import MusicTaggerCRNN
import audio_processor as ap
import pdb


def librosa_exists():
    try:
        __import__('librosa')
    except ImportError:
        return False
    else:
        return True


def main(net):
    ''' *WARNIING*
    This model use Batch Normalization, so the prediction
    is affected by batch. Use multiple, different data 
    samples together (at least 4) for reliable prediction.'''

    print('Running main() with network: %s and backend: %s' % (net, K._BACKEND))
    # setting
    audio_paths = ['data/bensound-cute.mp3',
                   'data/bensound-actionable.mp3',
                   'data/bensound-dubstep.mp3',
                   'data/bensound-thejazzpiano.mp3']
    melgram_paths = ['data/bensound-cute.npy',
                     'data/bensound-actionable.npy',
                     'data/bensound-dubstep.npy',
                     'data/bensound-thejazzpiano.npy']

    # prepare data like this
    melgrams = np.zeros((0, 1, 96, 1366))

    if librosa_exists:
        for audio_path in audio_paths:
            melgram = ap.compute_melgram(audio_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
    else:
        for melgram_path in melgram_paths:
            melgram = np.load(melgram_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)

    # load model like this
    if net == 'cnn':
        model = MusicTaggerCNN(weights='msd', include_top=False)
    elif net == 'crnn':
        model = MusicTaggerCRNN(weights='msd', include_top=False)
    # predict the tags like this
    print('Predicting features...')
    start = time.time()
    features = model.predict(melgrams)
    print features[:, :10]
    return

if __name__ == '__main__':

    networks = ['cnn', 'crnn']
    for net in networks:
        main(net)
