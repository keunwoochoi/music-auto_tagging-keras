# compact_cnn

20 Jan 2017, Keunwoo Choi

## What is this?
* A compact cnn - 5 layers, batch normalizationm, ELU, 32 feature maps for each conv layers.
* AUC is about 0.849 for the tagging task.

## Is it good?
![alt text](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/compact_cnn/benchmark/result_svm.png "results")

(pre-trained features + SVM)

More details coming soon.

## Before you run it
* set `image_dim_ordering()` == `th`.
* It works on both tensorflow/theano backend. 
* install [kapre](https://github.com/keunwoochoi/kapre) by 
```
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ python setup.py install
```


## Running it
* See `main.py` for an example.
* It is not the most efficient implementation, but the easiest for me :) still it's not slow even for cpu-based inference.


## Note
Tested on Keras 1.2.1

