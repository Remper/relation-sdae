import tensorflow as tf
import gzip
from time import time
from random import shuffle, randrange
import numpy as np


class RSDAEDataGenerator:
    """A data generator class."""

    def __init__(self, embeddings, input_path, batch_size, max_epoch, report_period, whitelist=None, max_sent_size=64):
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.embeddings = embeddings
        self.input_path = input_path
        self.max_sent_size = max_sent_size
        self.report_period = report_period
        self.whitelist = whitelist if whitelist and len(whitelist) > 0 else None

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
        lines = 0
        firstline = None
        backup_batch = ([], [])
        for self._epoch in range(self.max_epoch):
            with self._open_reader() as reader:
                batch = []
                targets = []
                for raw_line in reader:
                    raw_line = raw_line.strip()
                    lines += 1
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
                        if pos in self._noun_pos_tags and id is not self.embeddings.unk and not (self.whitelist and id not in self.whitelist):
                            nouns.append(len(words))
                        # Filter out some specials, there is a possible bug
                        if pos in ["CD", ":"]:
                            continue
                        if pos in [",", "''"]:
                            specials += 1
                        words.append(id)
                        if len(words) >= self.max_sent_size:
                            break

                    # Filtering out the sentence if too many special characters, too short or doesn't contain nouns
                    if len(words) < 3 or float(specials) / len(words) > 0.45 or len(nouns) < 1:
                        skipped_special += 1
                        continue

                    shuffle(nouns)

                    def replace_and_add(batch, targets, words, id):
                        words = words.copy()
                        targets.append(words[id])
                        words[id] = self.embeddings.noun
                        batch.append(words)

                    def replace_and_add_total(batch, targets, words, id):
                        target = words[id]
                        words = [self.embeddings.noun if target == ele else ele for ele in words]
                        targets.append(target)
                        batch.append(words)

                    # Replacing random noun, producing targets
                    replace_and_add_total(batch, targets, words, nouns[0])
                    # Save the rest for later
                    for noun_id in nouns[1:]:
                        replace_and_add_total(backup_batch[0], backup_batch[1], words, noun_id)

                    if firstline is None:
                        firstline = raw_line

                    # If we have sufficient backlog of other nouns in the same sentence, start randomly pick from them
                    if len(backup_batch[0]) >= self.batch_size * 10:
                        while len(batch) < self.batch_size:
                            id = randrange(0, len(backup_batch[0]))
                            batch.append(backup_batch[0].pop(id))
                            targets.append(backup_batch[1].pop(id))

                    # Once we have full batch — generate it
                    if len(batch) == self.batch_size:
                        batches += 1
                        inputs, inputs_length = self.prepare_batch(batch)
                        if batches % self.report_period == 0:
                            print("-" * 60)
                            print("Batch prep time: %.3fs, lines: %.2fm" % (time() - timestamp, float(lines) / 1000000))
                            print("  Inputs: %s" % " ".join([self.embeddings.inv_dictionary[ele] for ele in inputs[0, :inputs_length[0]]]))
                            print("  Target: %s" % self.embeddings.inv_dictionary[targets[0]])
                            print("  Raw line: %s" % firstline)
                            print("  Skipped sentences: %d" % skipped_special)
                            print("-" * 60)
                        yield inputs, inputs_length, targets
                        firstline = None
                        timestamp = time()
                        batch = []
                        targets = []
