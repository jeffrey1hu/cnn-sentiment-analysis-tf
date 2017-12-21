# cnn-text-sentiment-analysis-tf
In this project we will implement a movie rating Sentiment (Positive/Negative) Classifier with CNN using TensorFlow.
This cnn model is implement under the guide of the great blog post on [CNN classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).


## Dataset
The existing data set is the [Moive review data from Rotten Tomatoes](http://www.cs.cornell.edu/people/pabo/movie-review-data/) which is pretty small but convenient to tune the model under CPUs.
The adaption of other dataset (such as SST) is under development.

## requirements
- Python 3
- Tensorflow > 0.12
- Numpy

## Basic Usage
* Set the hyper-parameters in `config.py`.
* Then run with existing dataset
```shell
python train.py
```

## TO DO
Some updates will be published soon.
- add TensorBoard visualization
- add learning rate exponential decay to enhence generalization
- Initialize the embeddings with pre-trained word vectors (word2vec, glove)
- some way to prevent overfitting (l2 regularization, increase dropout rate..)