# music-auto_tagging-keras

## The prerequisite
* You need [`keras`](http://keras.io) and [`librosa`](http://librosa.github.io/librosa/) to execute `example.py`.
* You need `keras` to execute `example_without_librosa.py`.
* The input data shape is `(None, channel, height, width)`, i.e. following theano convention. If you're using tensorflow as your backend, you should check out `~/.keras/keras.json` if `image_dim_ordering` is set to `th`, i.e.
```json
"image_dim_ordering": "th",
```

### What happens? & Usage
```bash
$ python example.py
```
(or `$ python example_without_librosa.py`),

After a summary of convnet, the result will be printed:
``` bash
data/bensound-cute.mp3
[('jazz', 0.32834091782569885), ('folk', 0.17664788663387299), ('instrumental', 0.1569863110780716), ('guitar', 0.10749899595975876), ('acoustic', 0.08458312600851059), ('female vocalists', 0.06621211022138596), ('indie', 0.0627480000257492), ('chillout', 0.05570304021239281), ('rock', 0.04766707867383957), ('pop', 0.04348916560411453)]

data/bensound-actionable.mp3
[('rock', 0.4575064182281494), ('classic rock', 0.3454620838165283), ('punk', 0.23092204332351685), ('60s', 0.11653172969818115), ('70s', 0.11155932396650314), ('hard rock', 0.10467251390218735), ('indie', 0.1011115238070488), ('80s', 0.09881759434938431), ('alternative', 0.0769491195678711), ('Progressive rock', 0.0754147469997406)]

data/bensound-dubstep.mp3
[('Hip-Hop', 0.1726689487695694), ('rock', 0.10726829618215561), ('electronic', 0.10054843127727509), ('female vocalists', 0.07955039292573929), ('pop', 0.07343248277902603), ('alternative', 0.05530229210853577), ('indie', 0.04597167670726776), ('rnb', 0.04486352205276489), ('80s', 0.031885139644145966), ('90s', 0.02957077883183956)]

data/bensound-thejazzpiano.mp3
[('jazz', 0.9577991366386414), ('instrumental', 0.11406592279672623), ('guitar', 0.03199296444654465), ('rock', 0.024645458906888962), ('blues', 0.02134867012500763), ('chillout', 0.013597516342997551), ('easy listening', 0.013440641574561596), ('folk', 0.013292261399328709), ('oldies', 0.011634128168225288), ('country', 0.011065035127103329)]
```

## Files
* [example.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example.py): example
* [example_without_librosa.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example_without_librosa.py): example that doesn't require librosa because it uses pre-computed mel-spectrograms. If you want to test your own music files, you will anyway need to install [`librosa`](http://librosa.github.io/librosa/).
* [convnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/convnet.py): build and compile a convnet model
* [audio_processor.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py): compute mel-spectrogram using librosa
* Under [data/](https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data),
  - four .mp3 files: test files
  - four .npy files: pre-computed melgram for those who don't want to install librosa
  - [weights_best.hdf5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/weights_best.hdf5): pre-trained weights so that you don't need to train by yourself.

### The model
AUC score of 0.8454 for 50 music tags, trained on Million-Song Dataset.
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
is like this. A 'Narrow' version, which is quite nice considering a wide and very deep convnet shows AUC of 0.8595.
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
More info: [on this paper](https://arxiv.org/abs/1606.00298)


### Credits
* Please cite [this paper](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C), *Automatic Tagging using Deep Convolutional Neural Networks*, Keunwoo Choi, George Fazekas, Mark Sandler
17th International Society for Music Information Retrieval Conference, New York, USA, 2016

* Test music items are from [http://www.bensound.com](http://www.bensound.com).
