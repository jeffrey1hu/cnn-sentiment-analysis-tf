import logging
class CNNConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Data loading params
    dev_sample_percentage = .1
    positive_data_file = './data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = './data/rt-polaritydata/rt-polarity.neg'

    # Model Hyperparameters
    embed_size = 50
    filter_sizes = [3, 4, 5]
    num_filters = 128
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.0

    # Training parameters
    batch_size = 64
    n_epochs = 200
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5
    lr = 0.001

    # Other Parameters
    allow_soft_placement = True
    log_device_placement = False

    def __init__(self, max_sequence_length, n_classes):
        self.max_sequence_length = max_sequence_length
        self.n_classes = n_classes


class RNNConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Data loading params
    dev_sample_percentage = .1
    positive_data_file = './data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = './data/rt-polaritydata/rt-polarity.neg'

    # Model Hyperparameters
    hidden_size = 200
    keep_prob = 0.5
    embed_size = 50

    # Training parameters
    batch_size = 64
    n_epochs = 200
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5
    lr = 0.0001

    # Other Parameters
    allow_soft_placement = True
    log_device_placement = False

    def __init__(self, max_sequence_length, n_classes):
        self.max_sequence_length = max_sequence_length
        self.n_classes = n_classes