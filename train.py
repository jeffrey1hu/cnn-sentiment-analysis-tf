import time
import os
import tensorflow as tf
from utils.dataset import MRDataset
from cnn_model import CNNModel
from config import Config

dataset = MRDataset(path='./data/rt-polaritydata', debug=False)
max_sequence_length = dataset.max_document_length
n_classes = dataset.n_class
vocab_size = len(dataset.vocab_processor.vocabulary_)

print("max sequence length in dataset is %d" % max_sequence_length)
print("num of classes is %d" % n_classes)
print("vocab size is %d" % vocab_size)

config = Config(max_sequence_length, n_classes)

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        init_embedding = tf.random_uniform([vocab_size, config.embed_size], -1.0, 1.0)
        model = CNNModel(config, init_embedding)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        dataset.vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        model.fit(sess, saver, dataset, checkpoint_dir)
