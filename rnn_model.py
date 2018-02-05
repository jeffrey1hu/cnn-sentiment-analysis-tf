import tensorflow as tf
from model import Model

def sentence_length(sequence_mask):
    """
    Args:
        sequence_mask: Bool tensor with shape -> [batch_size, q]

    Returns:
        tf.int32, [batch_size]

    """
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)


class RNNModel(Model):

    def add_embedding(self):
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            embedding_mat = tf.Variable(self.init_embeddings, name="embedding_mat")
            embedded_sentence = tf.nn.embedding_lookup(embedding_mat, self.input_placeholder)
        return embedded_sentence

    def add_prediction_op(self):
        # shape of embedded_sentence -> (None, max_length, embed_size)
        embedded_sentence = self.add_embedding()

        with tf.variable_scope("rnn"):
            mask = tf.not_equal(self.input_placeholder, 0)

            sentence_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)
            sentence_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)


            # shape of outputs_states -> [output_fw, output_bw] -> output_fw -> [batch_size, hidden_size]
            outputs, [(c1, h1), (c2, h2)] = tf.nn.bidirectional_dynamic_rnn(sentence_lstm_fw_cell,
                                                         sentence_lstm_bw_cell,
                                                         embedded_sentence,
                                                         sequence_length=sentence_length(mask),
                                                         dtype=tf.float32)
            # output_H -> [None, 2 * hidden_size]
            final_state  = tf.concat([h1, h2], axis=1)
            print("shape of outputs_states is ", h1.get_shape().as_list())
            print("shape of final state is ", final_state.get_shape().as_list())

        # add dropout
        with tf.variable_scope("dropout"):

            final_state_dropout = tf.nn.dropout(final_state, keep_prob=self.config.keep_prob)

        with tf.variable_scope("output"):
            W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.config.n_classes], initializer=tf.constant_initializer(0.1))
            scores = tf.nn.xw_plus_b(final_state_dropout, W, b, name="scores")
            # shape of scores -> (Batch_szie, n_class)
            predictions = tf.argmax(scores, 1, name="predictions")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.labels_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        return scores, predictions, accuracy
