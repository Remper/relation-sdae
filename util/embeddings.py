import tensorflow as tf
import numpy as np
import time
import gzip


class Embeddings(object):

    def __init__(self, dictionary, embeddings=None):
        self.dictionary = dictionary
        self.inv_dictionary = self._populate_inverse_dictionary(dictionary)
        self.dims = 300
        self.unk = self.dictionary["<unk>"]
        self.bos = self.dictionary["<bos>"]
        self.eos = self.dictionary["<eos>"]
        self.noun = self.dictionary["<noun>"]

        if embeddings is not None:
            # Static pretrained embeddings
            self.embeddings = tf.constant(embeddings, name='embeddings', dtype=tf.float32)
            # Trainable copy
            self.emb_variable = tf.get_variable('emb_variable', initializer=self.embeddings)
            self.dims = embeddings.shape[1]
        else:
            # Initialising embeddings randomly if not pretrained
            self.emb_variable = tf.get_variable('emb_variable', shape=[len(self.dictionary), self.dims])

    def __len__(self):
        return len(self.dictionary)

    def _populate_inverse_dictionary(self, dictionary):
        inv_dict = dict()
        for key in dictionary:
            inv_dict[dictionary[key]] = key

        return inv_dict

    def get(self, word):
        if word in self.dictionary:
            return self.dictionary[word]
        return self.unk

    @staticmethod
    def restore_from_embeddings(file):
        final_embeddings = list()
        embeddings = list()
        dictionary = dict()
        count = 0
        timestamp = time.time()
        with Embeddings.open(file) as reader:
            for line in reader:
                row = line.rstrip().split('\t')

                if row[0] not in dictionary:
                    dictionary[row[0]] = count
                embeddings.append(np.array([float(ele) for ele in row[1:]]))

                count += 1
                if count % 100000 == 0:
                    print("  %.2fm embeddings parsed (%.3fs)" % (float(count) / 1000000, time.time() - timestamp))
                    timestamp = time.time()
                    final_embeddings.append(np.vstack(embeddings))
                    del embeddings
                    embeddings = list()

        print("  %.2fm embeddings parsed (%.3fs)" % (float(count) / 1000000, time.time() - timestamp))
        final_embeddings.append(np.vstack(embeddings))
        del embeddings

        return Embeddings(dictionary, np.vstack(final_embeddings))

    @staticmethod
    def restore_from_file(file):
        dictionary = dict()
        with Embeddings.open(file) as reader:
            for line in reader:
                line = line.rstrip()
                if line not in dictionary:
                    dictionary[line] = len(dictionary)
        return Embeddings(dictionary)

    @staticmethod
    def open(file):
        if file.endswith('.gz'):
            return gzip.open(file, 'rt', encoding="utf-8")
        return open(file, 'r', encoding="utf-8")
