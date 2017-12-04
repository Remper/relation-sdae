import tensorflow as tf
import gzip
from time import time
from random import choice
import numpy as np


class RSDAEDataGenerator:
    """A data generator class."""

    def __init__(self, embeddings, input_path, batch_size, max_epoch, report_period, max_sent_size=64):
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.embeddings = embeddings
        self.input_path = input_path
        self.max_sent_size = max_sent_size
        self.report_period = report_period

        self._noun_pos_tags = ["NN", "NNS"]
        self._proper_noun_pos_tags = ["NNP", "NNPS"]
        self._reader = None
        self._open_reader()
        self._epoch = 0

    def _open_reader(self):
        return gzip.open(self.input_path, 'rt', encoding="utf-8")

    @staticmethod
    def get_data_tensor(data_path):
        data = tf.contrib.data.tf.data.TFRecordDataset(data_path)
        return data.map(RSDAEDataGenerator._parse_tfrecord)

    @staticmethod
    def _parse_tfrecord(example_proto):
        features = {"text": tf.FixedLenSequenceFeature([241], tf.int64, allow_missing=True),
                    "pos": tf.FixedLenSequenceFeature([241], tf.int64, allow_missing=True)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["text"], parsed_features["pos"]

    @property
    def progress(self):
        return self._epoch

    def map_words(self, word):
        word = word.split('@')
        if len(word) > 2:
            word = '@'
            pos = 'SYM'
        else:
            pos = word[1]
            word = word[0]
            if pos not in self._proper_noun_pos_tags:
                word = word.lower()
        return self.embeddings.get(word), pos

    def pad_batch(self, batch, batch_lengths):
        """
        Pad the batch with beginning of sentence and ending of sentence tokens, adjust lengths accordingly

        :param batch:
        :param batch_lengths:
        :return:
        """
        padded_batch = np.full((batch.shape[0], batch.shape[1] + 2), self.embeddings.bos)
        padded_batch[:, 1:-1] = batch
        for i in range(len(batch_lengths)):
            padded_batch[i, batch_lengths[i]+1] = self.embeddings.eos
        return padded_batch, batch_lengths + 2

    def prepare_batch(self, batch):
        max_length = 0
        lengths = []

        for sentence in batch:
            cur_len = len(sentence)
            if max_length < cur_len:
                max_length = cur_len

            lengths.append(cur_len)
        lengths = np.array(lengths)

        return np.array([sentence + [0]*(max_length - len(sentence)) for sentence in batch]), lengths

    def __iter__(self):
        timestamp = time()
        batches = 0
        skipped_special = 0
        firstline = None
        for self._epoch in range(self.max_epoch):
            with self._open_reader() as reader:
                batch = []
                replacements = []
                for raw_line in reader:
                    raw_line = raw_line.strip()
                    if len(raw_line) == 0:
                        continue
                    line = raw_line.split(" ")
                    # Skipping dot in the end of the sentence
                    if line[len(line)-1] == ".@.":
                        line.pop()

                    # Converting words into ids, keeping track of all nouns in the sentence
                    words = []
                    nouns = []
                    specials = 0
                    for word in line:
                        id, pos = self.map_words(word)
                        if pos in self._noun_pos_tags:
                            nouns.append(len(words))
                        if pos in ["CD", ",", "''", ":"]:
                            specials += 1
                        words.append(id)
                        if len(words) >= self.max_sent_size:
                            break

                    # Filtering out the sentence if too many special characters, too short or doesn't contain nouns
                    if float(specials) / len(words) > 0.45 or len(words) < 3 or len(nouns) == 0:
                        skipped_special += 1
                        continue

                    # Replacing random noun, rembembering where to reconstruct targets
                    replacement = choice(nouns)
                    replacements.append((len(batch), replacement, words[replacement]))
                    words[replacement] = self.embeddings.noun

                    if firstline is None:
                        firstline = raw_line
                    batch.append(words)

                    if len(batch) == self.batch_size:
                        batches += 1
                        inputs, inputs_length = self.prepare_batch(batch)
                        targets, targets_length = self.pad_batch(inputs, inputs_length)
                        for row, col, word in replacements:
                            targets[row][col+1] = word
                        if batches % self.report_period == 0:
                            print("-" * 60)
                            print("Batch prep time: %.3fs" % (time() - timestamp))
                            print(" Inputs: %s" % " ".join([self.embeddings.inv_dictionary[ele] for ele in inputs[0, :inputs_length[0]]]))
                            print(" Targets: %s" % " ".join([self.embeddings.inv_dictionary[ele] for ele in targets[0, :targets_length[0]]]))
                            print(" Raw line: %s" % firstline)
                            print(" Skipped sentences with a lot of special symbols: %d" % skipped_special)
                            print("-" * 60)
                        yield inputs, inputs_length, targets, targets_length
                        firstline = None
                        timestamp = time()
                        batch = []
                        replacements = []

    def sample(self, num_samples):
        sample_inds = np.random.permutation(len(self.data))[:num_samples]
        words_sample = [self.data[i] for i in sample_inds]
        inputs, inputs_length, targets, targets_length = (
            self.construct_data(words_sample))
        return inputs, inputs_length, targets, targets_length