# cnn-text-sentiment-analysis-tf
In this project we will implement a movie rating Sentiment (Positive/Negative) Classifier with CNN using TensorFlow.
The project is under the guide of the great blog post on [CNN classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).


## Dataset
The existing data set is the [Moive review data from Rotten Tomatoes](http://www.cs.cornell.edu/people/pabo/movie-review-data/) which is pretty small but convenient to tune the model under CPUs.
The adaption of other dataset (such as SST) is under development.

## requirements
- Python 2.7
- Tensorflow 1.3.0
- Numpy

## Basic Usage
* Set the hyper-parameters in `config.py`.
* Then run with existing dataset
```shell
python train.py
```

## TO DO
- add TensorBoard visualization
- add learning rate exponential decay to enhence generalization
- Initialize the embeddings with pre-trained word vectors (word2vec, glove)
- some way to prevent overfitting (l2 regularization, increase dropout rate..)
- add interactive evaluation

## References
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* author's [Theano code](https://github.com/yoonkim/CNN_sentence)
* Denny Britz's [Tensorflow implementation](https://github.com/dennybritz/cnn-text-classification-tf)