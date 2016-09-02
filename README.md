# music-auto_tagging-keras

## The prerequisite
* You need [`keras`](http://keras.io) to run `example.py`.
  * To use your own audio file, you need [`librosa`](http://librosa.github.io/librosa/).
* The input data shape is `(None, channel, height, width)`, i.e. following theano convention. If you're using tensorflow as your backend, you should check out `~/.keras/keras.json` if `image_dim_ordering` is set to `th`, i.e.
```json
"image_dim_ordering": "th",
```

### What happens? & Usage
```bash
$ python example.py
```

After a summary of the networks, the result will be printed:
``` bash
$ python example.py
Using Theano backend.
Running main() with network: rnn
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
zeropadding2d_1 (ZeroPadding2D)  (None, 1, 96, 1440)   0           zeropadding2d_input_1[0][0]
____________________________________________________________________________________________________
batchnormalization_1 (BatchNormal(None, 1, 96, 1440)   2880        zeropadding2d_1[0][0]
____________________________________________________________________________________________________
ConvBNEluDr (Sequential)         (None, 128, 1, 15)    370560      batchnormalization_1[0][0]
____________________________________________________________________________________________________
permute_1 (Permute)              (None, 15, 128, 1)    0           ConvBNEluDr[1][0]
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 15, 128)       0           permute_1[0][0]
____________________________________________________________________________________________________
gru_1 (GRU)                      (None, 15, 32)        15456       reshape_1[0][0]
____________________________________________________________________________________________________
gru_2 (GRU)                      (None, 32)            6240        gru_1[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 32)            0           gru_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 50)            1650        dropout_5[0][0]
====================================================================================================
Total params: 396786
____________________________________________________________________________________________________
Loading weights of rnn...
Predicting...
Prediction is done. It took 11 seconds.
Printing top-15 tags for each track...
data/bensound-cute.mp3
[('jazz', '0.166'), ('indie', '0.136'), ('ambient', '0.124'), ('folk', '0.123'), ('electronic', '0.121')]
[('female vocalists', '0.117'), ('chillout', '0.107'), ('instrumental', '0.094'), ('acoustic', '0.081'), ('rock', '0.075')]
[('Mellow', '0.070'), ('pop', '0.069'), ('beautiful', '0.064'), ('alternative', '0.063'), ('electronica', '0.038')]

data/bensound-actionable.mp3
[('rock', '0.395'), ('classic rock', '0.208'), ('hard rock', '0.114'), ('80s', '0.103'), ('60s', '0.071')]
[('pop', '0.069'), ('70s', '0.067'), ('blues', '0.063'), ('punk', '0.061'), ('oldies', '0.052')]
[('alternative', '0.051'), ('country', '0.045'), ('indie', '0.041'), ('heavy metal', '0.032'), ('alternative rock', '0.030')]

data/bensound-dubstep.mp3
[('dance', '0.400'), ('electronic', '0.311'), ('pop', '0.189'), ('House', '0.104'), ('electro', '0.099')]
[('electronica', '0.065'), ('rock', '0.056'), ('female vocalists', '0.054'), ('80s', '0.045'), ('90s', '0.041')]
[('indie', '0.039'), ('Hip-Hop', '0.031'), ('alternative', '0.029'), ('party', '0.024'), ('rnb', '0.019')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.632'), ('blues', '0.092'), ('instrumental', '0.073'), ('folk', '0.038'), ('guitar', '0.031')]
[('rock', '0.020'), ('female vocalists', '0.020'), ('soul', '0.009'), ('experimental', '0.009'), ('oldies', '0.009')]
[('indie', '0.008'), ('acoustic', '0.007'), ('electronic', '0.007'), ('alternative', '0.007'), ('pop', '0.007')]

Running main() with network: cnn
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
batchnormalization_6 (BatchNormal(None, 1, 96, 1366)   2732        batchnormalization_input_1[0][0]
____________________________________________________________________________________________________
sequential_3 (Sequential)        (None, 256, 1, 1)     850368      batchnormalization_6[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 256)           0           sequential_3[1][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            12850       flatten_1[0][0]
====================================================================================================
Total params: 865950
____________________________________________________________________________________________________
Loading weights of cnn...
Predicting...
Prediction is done. It took 14 seconds.
Printing top-15 tags for each track...
data/bensound-cute.mp3
[('jazz', '0.393'), ('instrumental', '0.183'), ('folk', '0.143'), ('guitar', '0.109'), ('female vocalists', '0.067')]
[('acoustic', '0.062'), ('chillout', '0.061'), ('indie', '0.045'), ('electronic', '0.044'), ('rock', '0.041')]
[('pop', '0.040'), ('Mellow', '0.035'), ('chill', '0.034'), ('blues', '0.033'), ('ambient', '0.032')]

data/bensound-actionable.mp3
[('rock', '0.473'), ('classic rock', '0.381'), ('punk', '0.198'), ('60s', '0.127'), ('hard rock', '0.123')]
[('indie', '0.104'), ('70s', '0.102'), ('Progressive rock', '0.088'), ('alternative', '0.080'), ('80s', '0.080')]
[('blues', '0.076'), ('pop', '0.059'), ('indie rock', '0.056'), ('alternative rock', '0.039'), ('heavy metal', '0.033')]

data/bensound-dubstep.mp3
[('Hip-Hop', '0.139'), ('rock', '0.111'), ('electronic', '0.089'), ('pop', '0.088'), ('female vocalists', '0.072')]
[('alternative', '0.050'), ('rnb', '0.049'), ('80s', '0.044'), ('indie', '0.042'), ('90s', '0.038')]
[('soul', '0.035'), ('electronica', '0.027'), ('dance', '0.023'), ('hard rock', '0.022'), ('experimental', '0.020')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.964'), ('instrumental', '0.128'), ('guitar', '0.040'), ('rock', '0.026'), ('blues', '0.020')]
[('chillout', '0.019'), ('easy listening', '0.014'), ('folk', '0.014'), ('experimental', '0.013'), ('female vocalists', '0.013')]
[('electronic', '0.012'), ('alternative', '0.011'), ('oldies', '0.011'), ('Progressive rock', '0.010'), ('soul', '0.009')]
```

## Files
* [example.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example.py): example
* [convnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/convnet.py): build and compile a convnet model
* [recurrentnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/recurrentnet.py): build and compile a recurrentnet model
* [audio_processor.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py): compute mel-spectrogram using librosa
* Under [data/](https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data),
  - four .mp3 files: test files
  - four .npy files: pre-computed melgram for those who don't want to install librosa
  - [cnn_weights_best.hdf5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_best.hdf5): pre-trained weights so that you don't need to train by yourself.
  - [rnn_weights_best.hdf5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/rnn_weights_best.hdf5): similar but it uses conv+rnn. 

### The model
convnet: AUC score of 0.8454 for 50 music tags, trained on Million-Song Dataset.
rnn: AUC score: 0.8xx ..(it is currently learning).
The tags are...
```python
['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 
'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 
'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock',
'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening',
'sexy', 'catchy', 'funk', 'electro' ,'heavy metal', 'Progressive rock',
'60s', 'rnb', 'indie pop', 'sad', 'House', 'happy']
```

### The convnet
is like this. A 'Narrow' version of the convnet in my paper, which is quite nice considering a wide and very deep convnet shows AUC of 0.8595.
```bash
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 32, 96, 1366)  320         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
batchnormalization_2 (BatchNormal(None, 32, 96, 1366)  64          convolution2d_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 32, 96, 1366)  0           batchnormalization_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 48, 341)   0           elu_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 48, 341)   0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 128, 48, 341)  36992       dropout_1[0][0]
____________________________________________________________________________________________________
batchnormalization_3 (BatchNormal(None, 128, 48, 341)  256         convolution2d_2[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 128, 48, 341)  0           batchnormalization_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 128, 24, 85)   0           elu_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128, 24, 85)   0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 128, 24, 85)   147584      dropout_2[0][0]
____________________________________________________________________________________________________
batchnormalization_4 (BatchNormal(None, 128, 24, 85)   256         convolution2d_3[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 128, 24, 85)   0           batchnormalization_4[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 128, 12, 21)   0           elu_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128, 12, 21)   0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 192, 12, 21)   221376      dropout_3[0][0]
____________________________________________________________________________________________________
batchnormalization_5 (BatchNormal(None, 192, 12, 21)   384         convolution2d_4[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 192, 12, 21)   0           batchnormalization_5[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 192, 4, 4)     0           elu_4[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 192, 4, 4)     0           maxpooling2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 256, 4, 4)     442624      dropout_4[0][0]
____________________________________________________________________________________________________
batchnormalization_6 (BatchNormal(None, 256, 4, 4)     512         convolution2d_5[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 256, 4, 4)     0           batchnormalization_6[0][0]
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 256, 1, 1)     0           elu_5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 256, 1, 1)     0           maxpooling2d_5[0][0]
====================================================================================================
Total params: 850368
____________________________________________________________________________________________________
```
* More info: [on this paper](https://arxiv.org/abs/1606.00298), or [blog post](https://keunwoochoi.wordpress.com/2016/06/02/paper-is-out-automatic-tagging-using-deep-convolutional-neural-networks/).
* Also please take a look on the [slide](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/slide-ismir-2016.pdf) at ismir 2016. It includes some results that are not in the paper.

#### The rnn
will update soon.

### Credits
* Please cite [this paper](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C), *Automatic Tagging using Deep Convolutional Neural Networks*, Keunwoo Choi, George Fazekas, Mark Sandler
17th International Society for Music Information Retrieval Conference, New York, USA, 2016

* Test music items are from [http://www.bensound.com](http://www.bensound.com).
