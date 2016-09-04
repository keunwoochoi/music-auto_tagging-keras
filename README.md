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
$ python example.py
Running main() with network: cnn and backend: tensorflow
Loading weights of cnn...
Predicting...
Prediction is done. It took 3 seconds.
Printing top-15 tags for each track...
data/bensound-cute.mp3
[('folk', '0.222'), ('pop', '0.166'), ('jazz', '0.160'), ('female vocalists', '0.092'), ('acoustic', '0.075')]
[('rock', '0.070'), ('easy listening', '0.059'), ('indie', '0.055'), ('Mellow', '0.051'), ('beautiful', '0.036')]
[('alternative', '0.035'), ('soul', '0.034'), ('guitar', '0.033'), ('country', '0.032'), ('chillout', '0.027')]

data/bensound-actionable.mp3
[('rock', '0.592'), ('classic rock', '0.245'), ('pop', '0.119'), ('alternative', '0.109'), ('punk', '0.086')]
[('indie', '0.083'), ('80s', '0.076'), ('hard rock', '0.073'), ('female vocalists', '0.062'), ('indie rock', '0.051')]
[('alternative rock', '0.048'), ('blues', '0.047'), ('70s', '0.045'), ('90s', '0.039'), ('60s', '0.036')]

data/bensound-dubstep.mp3
[('electronic', '0.313'), ('Hip-Hop', '0.160'), ('electro', '0.116'), ('rock', '0.107'), ('pop', '0.085')]
[('dance', '0.078'), ('electronica', '0.077'), ('alternative', '0.064'), ('female vocalists', '0.047'), ('rnb', '0.047')]
[('indie', '0.035'), ('sexy', '0.031'), ('alternative rock', '0.031'), ('00s', '0.027'), ('hard rock', '0.024')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.799'), ('instrumental', '0.420'), ('guitar', '0.042'), ('blues', '0.028'), ('rock', '0.023')]
[('Progressive rock', '0.021'), ('easy listening', '0.020'), ('experimental', '0.018'), ('oldies', '0.013'), ('chillout', '0.009')]
[('60s', '0.009'), ('alternative', '0.009'), ('folk', '0.009'), ('classic rock', '0.007'), ('indie', '0.007')]

Running main() with network: rnn and backend: tensorflow
Loading weights of rnn...
Predicting...
Prediction is done. It took 8 seconds.
Printing top-15 tags for each track...
data/bensound-cute.mp3
[('jazz', '0.167'), ('female vocalists', '0.165'), ('folk', '0.145'), ('pop', '0.117'), ('soul', '0.110')]
[('rock', '0.071'), ('acoustic', '0.057'), ('easy listening', '0.055'), ('country', '0.053'), ('oldies', '0.049')]
[('Mellow', '0.045'), ('blues', '0.045'), ('indie', '0.043'), ('beautiful', '0.032'), ('chillout', '0.031')]

data/bensound-actionable.mp3
[('rock', '0.480'), ('classic rock', '0.389'), ('hard rock', '0.216'), ('blues', '0.085'), ('70s', '0.074')]
[('80s', '0.071'), ('heavy metal', '0.053'), ('alternative', '0.040'), ('Progressive rock', '0.040'), ('60s', '0.032')]
[('alternative rock', '0.029'), ('punk', '0.025'), ('pop', '0.024'), ('guitar', '0.022'), ('90s', '0.017')]

data/bensound-dubstep.mp3
[('electronic', '0.513'), ('electro', '0.222'), ('dance', '0.166'), ('electronica', '0.134'), ('House', '0.098')]
[('indie', '0.087'), ('rock', '0.086'), ('pop', '0.055'), ('alternative', '0.054'), ('Hip-Hop', '0.044')]
[('experimental', '0.042'), ('indie rock', '0.033'), ('female vocalists', '0.024'), ('00s', '0.024'), ('party', '0.023')]

data/bensound-thejazzpiano.mp3
[('jazz', '0.915'), ('instrumental', '0.043'), ('female vocalists', '0.018'), ('guitar', '0.017'), ('easy listening', '0.014')]
[('blues', '0.013'), ('chillout', '0.008'), ('rock', '0.008'), ('Mellow', '0.007'), ('soul', '0.006')]
[('funk', '0.005'), ('chill', '0.005'), ('folk', '0.004'), ('pop', '0.004'), ('ambient', '0.004')]

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

* More info - CNN: 
  * [on this paper](https://arxiv.org/abs/1606.00298), or [blog post](https://keunwoochoi.wordpress.com/2016/06/02/paper-is-out-automatic-tagging-using-deep-convolutional-neural-networks/).
  * Also please take a look on the [slide](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/slide-ismir-2016.pdf) at ismir 2016. It includes some results that are not in the paper.
* More info - RNN:
  * Paper/slide coming soon.

### Credits
* Please cite [this paper](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C), *Automatic Tagging using Deep Convolutional Neural Networks*, Keunwoo Choi, George Fazekas, Mark Sandler
17th International Society for Music Information Retrieval Conference, New York, USA, 2016

* Test music items are from [http://www.bensound.com](http://www.bensound.com).
