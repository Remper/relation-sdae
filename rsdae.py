"""Sequential autoencoder implementation."""
import tensorflow as tf
from tensorflow.contrib import seq2seq


class RSDAE(object):
    def __init__(self, cell, embeddings, time_major=False):
        """
        Args:
            cell: An RNNCell object
            embeddings: An Embeddings object
        """

        self.embeddings = embeddings.emb_variable
        self.vocabulary_size = len(embeddings)
        self.cell = cell
        self.time_major = time_major

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
        return tf.nn.embedding_lookup(self.embeddings, inputs)

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
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            decoder_outputs, which is a float32
            (batch, max_len, cell.output_size) tensor that contains
            the cell's hidden state per time step
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            helper = seq2seq.TrainingHelper(self._lookup(targets), targets_length, time_major=self.time_major)
            decoder = seq2seq.BasicDecoder(self.cell, helper, encoder_state)
            outputs, _, _ = seq2seq.dynamic_decode(decoder, output_time_major=self.time_major)
            decoder = tf.layers.dense(outputs.rnn_output, self.vocabulary_size, use_bias=False)
        return decoder

    def loss(self, decoder_outputs, targets, targets_length):
        """
        Args:
            decoder_outputs: A return value of decode_train function
            output_fn: A function that projects a vector with length
                cell.output_size into a vector with length vocab_size
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

    def decode_inference(self, encoder_state, output_fn, vocab_size,
                         bos_id, eos_id, max_length, scope='decoder', reuse=None):
        """
        Args:
            cell: An RNNCell object
            embeddings: An embedding matrix with shape
                (vocab_size, word_dim)
            encoder_state: A tensor that contains the encoder state;
                its shape should match that of cell.zero_state
            output_fn: A function that projects a vector with length
                cell.output_size into a vector with length vocab_size;
                please beware of the scope, since it will be called inside
                'scope/rnn' scope
            vocab_size: The size of a vocabulary set
            bos_id: The ID of the beginning-of-sentence symbol
            eos_id: The ID of the end-of-sentence symbol
            max_length: The maximum length of a generated sentence;
                it stops generating words when this number of words are
                generated and <EOS> is not appeared till then
            scope: A VariableScope object of a string which indicates
                the scope
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            generated, which is a float32 (batch, <=max_len)
            tensor that contains IDs of generated words
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer(),
                               reuse=reuse):
            decoder_fn = seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn, encoder_state=encoder_state,
                embeddings=self.embeddings, start_of_sequence_id=bos_id,
                end_of_sequence_id=eos_id, maximum_length=max_length,
                num_decoder_symbols=vocab_size)
            generated_logits, _, _ = seq2seq.dynamic_rnn_decoder(
                cell=self.cell, decoder_fn=decoder_fn, time_major=False, scope='rnn')
        generated = tf.argmax(generated_logits, axis=2)
        return generated
