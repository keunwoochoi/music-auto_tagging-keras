# music-auto_tagging-keras

### The prerequisite
* You need [`keras`](http://keras.io) to run `example.py`.
  * To use your own audio file, you need [`librosa`](http://librosa.github.io/librosa/).
* The input data shape is `(None, channel, height, width)`, i.e. following theano convention. If you're using tensorflow as your backend, you should check out `~/.keras/keras.json` if `image_dim_ordering` is set to `th`, i.e.
```json
"image_dim_ordering": "th",
```

### Structures

![alt text](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/imgs/diagrams.png "structures")

##### ConvNet 
 * 5-layer 2D Convolutions
 * num_parameter: 865,950
 * AUC score of 0.8454

(FYI: with 3M parameter, a deeper ConvNet showed 0.8595 AUC.)

##### RecurrentNet
 * 4-layer 2D Convolutions + 2 GRU 
 * num_parameter: 396,786
 * AUC score: 0.8xx ..(it is currently learning).

### How was it trained?
 * Using 29.1s music files in [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/)
 * Check out more details on [this paper](https://arxiv.org/abs/1606.00298)
 * The tags are...

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

### Which is better?
 * Training: ConvNet is faster than RecurrentNet (wall-clock time)
 * Prediction: ConvNet > RecurrentNet
 * Memory Usage: RecurrentNet have smaller number of trainable parameters. Actually you can even decreases the number of feature maps. The RecurrentNet still works quite well in the case - i.e., the current setting is a little bit rich (or redundant). With ConvNet, you will see the performance decrease if you reduce down the parameters. 

Therefore, if you just wanna use the pre-trained weights, use ConvNet. If you wanna train by yourself, it's up to you. I would use RecurrentNet after downsize it to, like, 0.2M parameters (then the training time would be similar to ConvNet) in general.

### Usage
```bash
$ python example.py
```
Please take a look on the codes, it's pretty simple.

### Result

After a summary of the networks, the result will be printed:
``` bash
$ python example.py
Using Theano backend.
Running main() with network: cnn and backend: theano
Loading weights of cnn...
Predicting...
Prediction is done. It took 6 seconds.
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

Running main() with network: rnn and backend: theano
Loading weights of rnn...
Predicting...
Prediction is done. It took 9 seconds.
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

```

### Files
* [example.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example.py): example
* [convnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/convnet.py): build and compile a convnet model
* [recurrentnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/recurrentnet.py): build and compile a recurrentnet model
* [audio_processor.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py): compute mel-spectrogram using librosa
* Under [data/](https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data),
  - four .mp3 files: test files
  - four .npy files: pre-computed melgram for those who don't want to install librosa
  - [cnn_weights_best.hdf5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_best.hdf5): pre-trained weights so that you don't need to train by yourself.
  - [rnn_weights_best.hdf5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/rnn_weights_best.hdf5): similar but it uses conv+rnn. 


### And...

* More info: [on this paper](https://arxiv.org/abs/1606.00298), or [blog post](https://keunwoochoi.wordpress.com/2016/06/02/paper-is-out-automatic-tagging-using-deep-convolutional-neural-networks/).
* Also please take a look on the [slide](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/slide-ismir-2016.pdf) at ismir 2016. It includes some results that are not in the paper.

### Credits
* Please cite [this paper](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C), *Automatic Tagging using Deep Convolutional Neural Networks*, Keunwoo Choi, George Fazekas, Mark Sandler
17th International Society for Music Information Retrieval Conference, New York, USA, 2016

* Test music items are from [http://www.bensound.com](http://www.bensound.com).
