import tensorflow as tf
from model import Model


class CNNModel(Model):

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
