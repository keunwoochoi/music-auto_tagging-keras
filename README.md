# Music Auto-Tagger
Music auto-tagger using keras

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
 * AUC score of 0.8654

(FYI: with 3M parameter, a deeper ConvNet showed 0.8595 AUC.)

##### RecurrentNet
 * 4-layer 2D Convolutions + 2 GRU 
 * num_parameter: 396,786
 * AUC score: 0.8662

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

Therefore, if you just wanna use the pre-trained weights, use ConvNet. If you wanna train by yourself, it's up to you. I would use RecurrentNet after downsizing it to, like, 0.2M parameters (then the training time would be similar to ConvNet) in general. To reduce the size, change `nums_feat_maps` under `get_convBNeluMPdrop` in `recurrentnet.py`.

### Usage
```bash
$ python example.py
```
Please take a look on the codes, it's pretty simple.

### Result

``` bash
python example.py
Using Theano backend.
Running main() with network: cnn and backend: theano
Loading weights of cnn...
Predicting...
Prediction is done. It took 7 seconds.
Printing top-10 tags for each track...
data/bensound-cute.mp3
[('folk', '0.205'), ('jazz', '0.173'), ('pop', '0.153'), ('female vocalists', '0.103'), ('acoustic', '0.066')]
[('easy listening', '0.064'), ('rock', '0.050'), ('indie', '0.047'), ('Mellow', '0.044'), ('instrumental', '0.038')]

data/bensound-actionable.mp3
[('rock', '0.589'), ('classic rock', '0.309'), ('blues', '0.108'), ('alternative', '0.099'), ('hard rock', '0.093')]
[('pop', '0.085'), ('punk', '0.071'), ('indie', '0.066'), ('60s', '0.062'), ('70s', '0.061')]

data/bensound-dubstep.mp3
[('electronic', '0.262'), ('Hip-Hop', '0.141'), ('rock', '0.111'), ('pop', '0.101'), ('electro', '0.084')]
[('dance', '0.073'), ('electronica', '0.061'), ('alternative', '0.059'), ('rnb', '0.046'), ('female vocalists', '0.036')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.767'), ('instrumental', '0.439'), ('guitar', '0.037'), ('rock', '0.027'), ('easy listening', '0.019')]
[('Progressive rock', '0.018'), ('experimental', '0.018'), ('blues', '0.017'), ('alternative', '0.012'), ('chillout', '0.012')]

Running main() with network: rnn and backend: theano
Loading weights of rnn...
Predicting...
Prediction is done. It took 12 seconds.
Printing top-10 tags for each track...
data/bensound-cute.mp3
[('jazz', '0.238'), ('folk', '0.179'), ('female vocalists', '0.154'), ('pop', '0.098'), ('acoustic', '0.075')]
[('instrumental', '0.060'), ('indie', '0.058'), ('soul', '0.058'), ('chillout', '0.054'), ('rock', '0.051')]

data/bensound-actionable.mp3
[('rock', '0.474'), ('classic rock', '0.388'), ('hard rock', '0.243'), ('blues', '0.097'), ('heavy metal', '0.067')]
[('70s', '0.065'), ('80s', '0.061'), ('Progressive rock', '0.044'), ('alternative', '0.041'), ('60s', '0.034')]

data/bensound-dubstep.mp3
[('electronic', '0.572'), ('electro', '0.230'), ('electronica', '0.166'), ('dance', '0.138'), ('House', '0.096')]
[('indie', '0.093'), ('rock', '0.085'), ('experimental', '0.066'), ('alternative', '0.061'), ('pop', '0.044')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.922'), ('instrumental', '0.039'), ('female vocalists', '0.018'), ('guitar', '0.018'), ('blues', '0.016')]
[('easy listening', '0.012'), ('rock', '0.007'), ('chillout', '0.006'), ('Mellow', '0.006'), ('soul', '0.006')]


```

### Files
* [example.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example.py): example
* [audio_convnet.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_convnet.py): build a convnet model
* [audio_conv_rnn.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_conv_rnn.py): build a recurrentnet model
* [audio_processor.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py): compute mel-spectrogram using librosa
* Under [data/](https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data),
  - four .mp3 files: test files
  - four .npy files: pre-computed melgram for those who don't want to install librosa
  - [cnn_weights_tensorflow.h5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_tensorflow.h5), [cnn_weights_theano.h5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_theano.h5): pre-trained weights so that you don't need to train by yourself.
  - [rnn_weights_tensorflow.h5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/rnn_weights_tensorflow.h5), [rnn_weights_theano.h5](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/rnn_weights_theano.h5): similar but it's for conv+rnn. 


### And...

* More info - CNN: 
  * [on this paper](https://arxiv.org/abs/1606.00298), or [blog post](https://keunwoochoi.wordpress.com/2016/06/02/paper-is-out-automatic-tagging-using-deep-convolutional-neural-networks/).
  * Also please take a look on the [slide](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/slide-ismir-2016.pdf) at ismir 2016. It includes some results that are not in the paper.
* More info - RNN:
  * [paper](https://arxiv.org/abs/1609.04243), or [blog post](https://keunwoochoi.wordpress.com/2016/09/15/paper-is-out-convolutional-recurrent-neural-networks-for-music-classification/)

### Reproduce the experiment
* [A repo for split setting](https://github.com/keunwoochoi/MSD_split_for_tagging/) for an identical setting of experiments in [two papers](#credits). 
* Audio file: find someone around you who happened to have the preview clips. or you have to crawl the files. I would recommend you to crawl your colleagues...

### Credits
* Convnet: [*Automatic Tagging using Deep Convolutional Neural Networks*](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C), Keunwoo Choi, George Fazekas, Mark Sandler
17th International Society for Music Information Retrieval Conference, New York, USA, 2016
* ConvRNN : [*Convolutional Recurrent Neural Networks for Music Classification*](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&sortby=pubdate&citation_for_view=ZrqdSu4AAAAJ:ULOm3_A8WrAC), Keunwoo Choi, George Fazekas, Mark Sandler, Kyunghyun Cho, arXiv:1609.04243, 2016

* Test music items are from [http://www.bensound.com](http://www.bensound.com).
