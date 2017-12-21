import numpy as np
from os.path import join as pjoin
from data_helpers import clean_str
from tensorflow.contrib import learn

class MRDataset:
    def __init__(self, path=None, debug=True):
        if not path:
            path = 'data/rt-polaritydata'
        self.positive_data_file = pjoin(path, 'rt-polarity.pos')
        self.negative_data_file = pjoin(path, 'rt-polarity.neg')
        self.x_text, self.y = self.load_data_and_labels(debug=debug)
        self.max_document_length = max([len(x.split(" ")) for x in self.x_text])
        self.n_class = self.y.shape[1]
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length)
        self.x = np.array(list(self.vocab_processor.fit_transform(self.x_text)))

    def load_data_and_labels(self, debug=True):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(self.positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(self.negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        if debug:
            positive_examples = positive_examples[:1000]
            negative_examples = negative_examples[:1000]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def get_train_and_dev_data(self, dev_sample_percentage):

        dev_sample_index = -1 * int(dev_sample_percentage * float(len(self.y)))
        x_train, x_dev = self.x[:dev_sample_index], self.x[dev_sample_index:]
        y_train, y_dev = self.y[:dev_sample_index], self.y[dev_sample_index:]
        return (x_train, y_train), (x_dev, y_dev)




