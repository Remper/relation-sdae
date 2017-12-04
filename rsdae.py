"""Sequential autoencoder implementation."""
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense


class RSDAE(object):
    def __init__(self, cell, embeddings, time_major=False):
        """
        Args:
            cell: An RNNCell object
            embeddings: An Embeddings object
        """

        self.embeddings = embeddings
        self.vocabulary_size = len(embeddings)
        self.cell = cell
        self.time_major = time_major

    def get_output_layer(self):
        return Dense(self.vocabulary_size, use_bias=False, name="decoder-output")

    def encode(self, inputs, inputs_length, scope='encoder'):
        """
        Args:
            cell: An RNNCell object
            embeddings: An embedding matrix with shape
                (vocab_size, word_dim) and with float32 type
            inputs: A int32 tensor with shape (batch, max_len), which
                contains word indices
            inputs_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch
            scope: A VariableScope object of a string which indicates
                the scope
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            sent_vec, which is a int32 tensor with shape
            (batch, cell.output_size) that contains sentence representations
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            _, sent_vec = tf.nn.dynamic_rnn(
                cell=self.cell, inputs=self._lookup(inputs), sequence_length=inputs_length,
                dtype=tf.float32, time_major=self.time_major)
        return sent_vec

    def _lookup(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings.emb_variable, inputs)

    def decode_train(self, encoder_state, targets, targets_length, scope='decoder'):
        """
        Args:
            encoder_state: A tensor that contains the encoder state;
                its shape should match that of cell.zero_state
            targets: A int32 tensor with shape (batch, max_len), which
                contains word indices; should start and end with
                the proper <BOS> and <EOS> symbol
            targets_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch
            scope: A VariableScope object of a string which indicates
                the scope

        Returns:
            decoder_outputs, which is a float32
            (batch, max_len, cell.output_size) tensor that contains
            the cell's hidden state per time step
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            helper = seq2seq.TrainingHelper(self._lookup(targets), targets_length, time_major=self.time_major)
            decoder = seq2seq.BasicDecoder(self.cell, helper, encoder_state, output_layer=self.get_output_layer())
            outputs, _, _ = seq2seq.dynamic_decode(decoder, output_time_major=self.time_major)
        return outputs.rnn_output, outputs.sample_id

    def loss(self, decoder_outputs, targets, targets_length):
        """
        Args:
            decoder_outputs: A return value of decode_train function
            targets: A int32 tensor with shape (batch, max_len), which
                contains word indices
            targets_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch

        Returns:
            loss, which is a scalar float32 tensor containing an average
            cross-entropy loss value
        """

        max_len = decoder_outputs.get_shape()[1].value
        if max_len is None:
            max_len = tf.shape(decoder_outputs)[1]
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=decoder_outputs)
        losses_mask = tf.sequence_mask(
            lengths=targets_length, maxlen=max_len,
            dtype=tf.float32)
        return tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)

    def decode_inference(self, encoder_state, scope='decoder-inference'):
        """
        Args:
            encoder_state: A tensor that contains the encoder state;
                its shape should match that of cell.zero_state
            scope: A VariableScope object of a string which indicates
                the scope

        Returns:
            generated, which is a float32 (batch, <=max_len)
            tensor that contains IDs of generated words
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            helper = seq2seq.GreedyEmbeddingHelper(
                self._lookup,
                tf.fill([tf.shape(encoder_state)[0]], self.embeddings.bos),
                self.embeddings.eos
            )
            decoder = seq2seq.BasicDecoder(self.cell, helper, encoder_state, output_layer=self.get_output_layer())
            outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=64)
            return outputs.sample_id
