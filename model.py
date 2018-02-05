"""
Base model whose methods will be reused or reimplement in RNNModel and CNNModel
"""
import numpy as np
import tensorflow as tf

from utils.general_utils import Progbar
from utils.data_helpers import get_minibatches


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, config, init_embeddings):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.init_embeddings = init_embeddings
        self.config = config
        self.add_placeholders()
        self.scores, self.predictions, self.accuracy = self.add_prediction_op()
        self.loss = self.add_loss_op(self.scores)
        self.train_op, self.grad_summary = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.max_sequence_length], name='input_x')
        self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.n_classes], name='input_y')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed = dict()
        feed[self.input_placeholder] = inputs_batch
        feed[self.labels_placeholder] = labels_batch
        feed[self.dropout_placeholder] = dropout
        return feed

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, scores):
        with tf.variable_scope("loss"):
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.labels_placeholder))
        return losses

    def add_training_op(self, loss):
        # train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        return train_op, grad_summaries_merged

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout_keep_prob)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, dataset):
        train_examples, dev_set =dataset.get_train_and_dev_data(self.config.dev_sample_percentage)

        prog = Progbar(target=1 + len(train_examples[1]) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        dev_precision = self.evaluate(sess, dev_set)
        print "- dev precision: {:.2f}".format(dev_precision * 100.0)
        return dev_precision

    def fit(self, sess, saver, dataset, checkpoint_dir):

        best_precision = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_precision = self.run_epoch(sess, dataset)
            if dev_precision > best_precision:
                best_precision = dev_precision
                if saver:
                    print "New best dev UAS! Saving model in ./data/weights/model.weights"
                    saver.save(sess, checkpoint_dir)
            print

    def evaluate(self, sess, dev_set):
        accs = []
        for dev_x, dev_y in get_minibatches(dev_set, self.config.batch_size, shuffle=False):
            feed = self.create_feed_dict(dev_x, labels_batch=dev_y)
            accuracy = sess.run([self.accuracy], feed_dict = feed)
            accs.append(accuracy)
        return np.mean(accs)