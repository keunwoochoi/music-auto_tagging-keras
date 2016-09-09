import time
import numpy as np
from keras import backend as K
from audio_convnet import AudioConvnet
from audio_conv_rnn import AudioConvRNN
import audio_processor as ap


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]


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

    tags = ['rock', 'pop', 'alternative', 'indie', 'electronic',
            'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
            'beautiful', 'metal', 'chillout', 'male vocalists',
            'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
            '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
            'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
            'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
            '70s', 'party', 'country', 'easy listening',
            'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
            'Progressive rock', '60s', 'rnb', 'indie pop',
            'sad', 'House', 'happy']

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
        model = AudioConvnet()
    elif net == 'rnn':
        model = AudioConvRNN()
    
    print('Loading weights of %s...' % net)
    model.load_weights('data/%s_weights_%s.h5' % (net, K._BACKEND))
    # predict the tags like this
    print('Predicting...')
    start = time.time()
    pred_tags = model.predict(melgrams)
    # print like this...
    print "Prediction is done. It took %d seconds." % (time.time()-start)
    print('Printing top-10 tags for each track...')
    for song_idx, audio_path in enumerate(audio_paths):
        sorted_result = sort_result(tags, pred_tags[song_idx, :].tolist())
        print(audio_path)
        print(sorted_result[:5])
        print(sorted_result[5:10])
        print(' ')
    return

if __name__ == '__main__':

    networks = ['cnn', 'rnn']
    for net in networks:
        main(net)
