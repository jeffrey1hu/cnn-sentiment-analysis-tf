import time
import os
import tensorflow as tf
from argparse import ArgumentParser
from utils.dataset import MRDataset
from cnn_model import CNNModel
from rnn_model import RNNModel
from config import CNNConfig, RNNConfig

dataset = MRDataset(path='./data/rt-polaritydata', debug=False)
max_sequence_length = dataset.max_document_length
n_classes = dataset.n_class
vocab_size = len(dataset.vocab_processor.vocabulary_)

print("max sequence length in dataset is %d" % max_sequence_length)
print("num of classes is %d" % n_classes)
print("vocab size is %d" % vocab_size)


def parse_args():
    parser = ArgumentParser(
        description=('Build a deep learning architecture, and train from the '
                     'provided dataset'))

    parser.add_argument('-m', '--model', choices=["lstm", "cnn"], default="cnn", help="Type of model to train.")

    return parser.parse_args()


def main(args):
    if args.model == 'lstm':
        config = RNNConfig(max_sequence_length, n_classes)
    else:
        config = CNNConfig(max_sequence_length, n_classes)

    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            init_embedding = tf.random_uniform([vocab_size, config.embed_size], -1.0, 1.0)
            if args.model == 'lstm':
                model = RNNModel(config, init_embedding)
            else:
                model = CNNModel(config, init_embedding)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.model, timestamp))
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


if __name__ == '__main__':
    main(parse_args())