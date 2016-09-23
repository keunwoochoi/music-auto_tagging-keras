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

#### MusicTaggerCNN
 * 5-layer 2D Convolutions
 * num_parameter: 865,950
 * AUC score of 0.8654
 * **WARNING** with keras >1.0.6, this model does not work properly.
 Please use MusicTaggerCRNN until it is updated!
(FYI: with 3M parameter, a deeper ConvNet showed 0.8595 AUC.)

##### MusicTaggerCRNN
 * 4-layer 2D Convolutions + 2 GRU 
 * num_parameter: 396,786
 * AUC score: 0.8662

### How was it trained?
 * Using 29.1s music files in [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/)
 * split setting: [A repo for split setting](https://github.com/keunwoochoi/MSD_split_for_tagging/) for an identical setting.
 * See [papers](#credits)
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
 * Training: `MusicTaggerCNN` is faster than `MusicTaggerCRNN` (wall-clock time)
 * Prediction: They are more or less the same. 
 * Memory Usage: `MusicTaggerCRNN` have smaller number of trainable parameters. Actually you can even decreases the number of feature maps. The `MusicTaggerCRNN` still works quite well in the case - i.e., the current setting is a little bit rich (or redundant). With `MusicTaggerCNN`, you will see the performance decrease if you reduce down the parameters. 

Therefore, if you just wanna use the pre-trained weights, use `MusicTaggerCNN`. If you wanna train by yourself, it's up to you. I would use `MusicTaggerCRNN` after downsizing it to, like, 0.2M parameters (then the training time would be similar to `MusicTaggerCNN`) in general. To reduce the size, change number of feature maps of convolution layers.

### Usage
```bash
$ python example_tagging.py
$ python example_feat_extract.py
```

### Result
(will be updated.)

### Files
#### Examples
* [example_tagging.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example_tagging.py): tagging example
* [example_feat_extract.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/example_feat_extract.py): feature extraction example

#### Models
* [music_tagger_cnn.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_cnn.py)
* [music_tagger_crnn.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py)

#### utility
* [audio_processor.py](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py): compute mel-spectrogram using librosa

#### weights
* Under [data/](https://github.com/keunwoochoi/music-auto_tagging-keras/tree/master/data),
  - [music_tagger_cnn_weights_tensorflow](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_cnn_weights_tensorflow.h5)
  - [music_tagger_crnn_weights_tensorflow](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_tensorflow.h5)
  - [music_tagger_cnn_weights_theano](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_cnn_weights_theano.h5)
  - [music_tagger_crnn_weights_theano](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_theano.h5)


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
