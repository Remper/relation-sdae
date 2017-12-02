"""Train the SDAE model."""
import argparse
import os
from time import time
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import layers, rnn, slim
from tensorflow.contrib.framework import get_or_create_global_step

from rsdae import RSDAE
from util.rsdae_data_generator import RSDAEDataGenerator
from util.embeddings import Embeddings


logging = tf.logging
logging.set_verbosity(logging.INFO)


def main():
    input_path = args.input
    embeddings_path = args.embeddings
    dict_path = args.dict
    output_path = args.output
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    sentence_dim = args.sentence_dim

    # Check the existence of needed directories
    if not os.path.exists(input_path) or not os.path.exists(embeddings_path):
        logging.error("Input or embeddings paths are incorrect")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            if dict_path:
                logging.info("Loading dictionary")
                embeddings = Embeddings.restore_from_file(dict_path)
            else:
                logging.info("Loading embeddings")
                embeddings = Embeddings.restore_from_embeddings(embeddings_path)

            logging.info('Initializing the data generator')
            data_generator = RSDAEDataGenerator(embeddings=embeddings, input_path=input_path, batch_size=batch_size,
                                                max_epoch=max_epoch)

            logging.info('Building the model...')
            # Placeholders
            inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                    name='inputs')
            inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                           name='inputs_length')
            targets = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                     name='targets')
            targets_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                            name='targets_length')

            # Defining cell and initialising RSDAE
            rnn_cell = rnn.GRUBlockCell(sentence_dim)
            rsdae = RSDAE(rnn_cell, embeddings)

            # Coupling together computation graph
            encoder_state = rsdae.encode(inputs=inputs, inputs_length=inputs_length, scope='encoder')
            decoder_outputs = rsdae.decode_train(encoder_state=encoder_state, targets=targets[:, :-1],
                                                 targets_length=targets_length - 1, scope='decoder')
            '''generated = rsdae.decode_inference(
                encoder_state=encoder_state, output_fn=output_fn,
                vocab_size=len(embeddings),
                bos_id=embeddings.dictionary['<eos>'],
                eos_id=embeddings.dictionary['<eos>'],
                max_length=50,
                scope='decoder', reuse=True)'''
            loss = rsdae.loss(decoder_outputs=decoder_outputs, targets=targets[:, 1:], targets_length=targets_length - 1)
            tf.summary.scalar("loss", loss)

            # Defining optimisation problem
            global_step = get_or_create_global_step()
            train_op = slim.optimize_loss(
                loss=loss, global_step=global_step, learning_rate=None,
                optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

            # Logging
            summary_writer = tf.summary.FileWriter(
                logdir=os.path.join(output_path, 'log'), graph=graph)
            summary = tf.summary.merge_all()

            logging.info('Initializing variables')
            timestamp = time()
            tf.get_variable_scope().set_initializer(tf.random_normal_initializer(mean=0.0, stddev=0.01))
            tf.global_variables_initializer().run()
            logging.info('Done in %.2fs' % (time() - timestamp))

            saver = tf.train.Saver(max_to_keep=20)

            logging.info('Starting training')
            for data_batch in data_generator:
                (inputs_v, inputs_length_v,
                 targets_v, targets_length_v) = data_batch
                summary_v, global_step_v, _ = sess.run(
                    fetches=[summary, global_step, train_op],
                    feed_dict={inputs: inputs_v,
                               inputs_length: inputs_length_v,
                               targets: targets_v,
                               targets_length: targets_length_v})
                summary_writer.add_summary(summary=summary_v,
                                           global_step=global_step_v)

                '''
                if global_step_v % 100 == 0:
                    logging.info('{} Iter #{}, Epoch {:.2f}'
                                 .format(datetime.now(), global_step_v,
                                         data_generator.progress))
                    num_samples = 2
                    (inputs_sample_v, inputs_length_sample_v,
                     targets_sample_v, targets_length_sample_v) = (
                        data_generator.sample(num_samples))
                    generated_v = sess.run(
                        fetches=generated,
                        feed_dict={inputs: inputs_sample_v,
                                   inputs_length: inputs_length_sample_v})
                    for i in range(num_samples):
                        logging.info('-' * 60)
                        logging.info('Sample #{}'.format(i))
                        inputs_sample_words = data_generator.ids_to_words(
                            inputs_sample_v[i][:inputs_length_sample_v[i]])
                        targets_sample_words = data_generator.ids_to_words(
                            targets_sample_v[i][1:targets_length_sample_v[i]])
                        generated_words = data_generator.ids_to_words(
                            generated_v[i])
                        if '<EOS>' in generated_words:
                            eos_index = generated_words.index('<EOS>')
                            generated_words = generated_words[:eos_index + 1]
                        logging.info('Input: {}'
                                     .format(' '.join(inputs_sample_words)))
                        logging.info('Target: {}'
                                     .format(' '.join(targets_sample_words)))
                        logging.info('Generated: {}'
                                     .format(' '.join(generated_words)))
                    logging.info('-' * 60)
                '''

                # Checkpointing
                if global_step_v % 500 == 0:
                    save_path = os.path.join(output_path, 'model.ckpt')
                    real_save_path = saver.save(sess=sess, save_path=save_path,
                                                global_step=global_step_v)
                    logging.info('Saved the checkpoint to: {}'
                                 .format(real_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the RSDAE')
    parser.add_argument('--input', required=True,
                        help='The directory with the input')
    parser.add_argument('--embeddings', required=False,
                        help='The file with pretrained embeddings')
    parser.add_argument('--dict', required=False,
                        help='The file with dictionary to start with randomly initialised embeddings')
    parser.add_argument('--output', required=True,
                        help='The directory where to save the output')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--sentence-dim', type=int, default=300,
                        help='The dimension of a sentence representation')

    args = parser.parse_args()
    main()
