import numpy as np
import tensorflow as tf
from model import Model
from utils.general_utils import Progbar
from utils.data_helpers import get_minibatches

class CNNModel(Model):
    def __init__(self, config, init_embeddings):
        self.init_embeddings = init_embeddings
        self.config = config
        self.add_placeholders()
        self.scores, self.predictions, self.accuracy = self.add_prediction_op()
        self.loss = self.add_loss_op(self.scores)
        self.train_op = self.add_training_op(self.loss)


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

    def add_embedding(self):
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            embedding_mat = tf.Variable(self.init_embeddings, name="embedding_mat")
            embedded_sentence = tf.nn.embedding_lookup(embedding_mat, self.input_placeholder)
            # conv2d operation expects a 4-dimensional tensor
            # with dimensions corresponding to batch, width, height and channel.
            embedded_sentence_expanded = tf.expand_dims(embedded_sentence, -1)
        return embedded_sentence_expanded

    def add_prediction_op(self):
        embedded_sentence_expanded = self.add_embedding()

        with tf.variable_scope("cnn"):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope('conv-maxpool-layer-%d' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.config.embed_size, 1, self.config.num_filters]
                    W = tf.get_variable("W", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("b", shape=[self.config.num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(
                        embedded_sentence_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Add nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    # ksize -> (batch_size, heigth, width, channel)
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.max_sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    # shape of pooled -> (batch_size, 1, 1, channel_size)
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # add dropout
        with tf.variable_scope("dropout"):
            h_pool_drop = tf.nn.dropout(h_pool_flat, self.dropout_placeholder)

        with tf.variable_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.config.n_classes], initializer=tf.constant_initializer(0.1))
            scores = tf.nn.xw_plus_b(h_pool_drop, W, b, name="scores")
            # shape of scores -> (Batch_szie, n_class)
            predictions = tf.argmax(scores, 1, name="predictions")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.labels_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        return scores, predictions, accuracy

    def add_loss_op(self, scores):
        with tf.variable_scope("loss"):
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.labels_placeholder))
        return losses

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

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