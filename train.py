"""Train the SDAE model."""
import argparse
import os
from time import time
from random import choice, random

import tensorflow as tf
from tensorflow.contrib import slim

from rsdae import RSDAE
from util.rsdae_data_generator import RSDAEDataGenerator
from util.embeddings import Embeddings
from util.evaluation import VRDEvaluation, EvaluationDataGenerator

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
    report_period = args.report
    evaluation_path = args.evaluation
    eval_enabled = args.eval

    logging.info("Initialised with parameters: %s" % vars(args))

    # Check the existence of needed directories
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if eval_enabled and not evaluation_path:
        logging.error("In order to enable eval — provide path to the evaluation set")
        return

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            if dict_path:
                logging.info("Loading dictionary")
                embeddings = Embeddings.restore_from_file(dict_path)
            else:
                logging.info("Loading embeddings")
                embeddings = Embeddings.restore_from_embeddings(embeddings_path)

            # Load evaluation dataset
            if evaluation_path:
                evaluation_train = VRDEvaluation.from_directory(evaluation_path, embeddings, include_train=True)
                print("Verbalized sample:")
                for i in range(5):
                    sentence, sentence_label = choice(evaluation_train.sentences)
                    sentence = " ".join([embeddings.inv_dictionary[ele] for ele in sentence])
                    print("  l: %s s: %s" % (embeddings.inv_dictionary[sentence_label], sentence))
            else:
                evaluation_train = VRDEvaluation.from_empty()

            logging.info('Initializing the data generator (eval enabled: %s)' % str(eval_enabled))
            if eval_enabled:
                data_generator = EvaluationDataGenerator(evaluation_train, batch_size=batch_size, max_epoch=max_epoch)
            else:
                data_generator = RSDAEDataGenerator(embeddings=embeddings, input_path=input_path, batch_size=batch_size,
                                                    report_period=report_period, max_epoch=max_epoch,
                                                    whitelist=evaluation_train.objects)

            logging.info('Building the model...')
            # Placeholders
            inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                    name='inputs')
            inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                           name='inputs_length')
            targets = tf.placeholder(dtype=tf.int32, shape=[None],
                                     name='targets')

            # Defining cell and initialising RSDAE
            forward_cell = tf.nn.rnn_cell.BasicLSTMCell(sentence_dim)
            backward_cell = tf.nn.rnn_cell.BasicLSTMCell(sentence_dim)
            rsdae = RSDAE((forward_cell, backward_cell), embeddings)

            # Coupling together computation graph
            encoder_state = rsdae.encode(inputs=inputs, inputs_length=inputs_length, scope='encoder')
            decoder_outputs = rsdae.decode_token(encoder_state=encoder_state, scope='decoder')
            result = tf.nn.top_k(decoder_outputs, k=10, name="result")
            loss = rsdae.loss_token(decoder_outputs=decoder_outputs, targets=targets)

            # Defining optimisation problem
            global_step = tf.train.get_or_create_global_step()
            train_op = slim.optimize_loss(
                loss=loss, global_step=global_step, learning_rate=None,
                optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

            # Logging
            summary_writer = tf.summary.FileWriter(
                logdir=os.path.join(output_path, 'log'), graph=graph)
            summary = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=10)

            latest_checkpoint = tf.train.latest_checkpoint(output_path)
            if latest_checkpoint is None:
                logging.info('Initializing variables')
                timestamp = time()
                tf.get_variable_scope().set_initializer(tf.random_normal_initializer(mean=0.0, stddev=0.01))
                tf.global_variables_initializer().run()
                logging.info('Done in %.2fs' % (time() - timestamp))
            else:
                logging.info('Restoring from checkpoint variables')
                timestamp = time()
                saver.restore(sess=sess, save_path=latest_checkpoint)
                logging.info('Done in %.2fs' % (time() - timestamp))

            logging.info('Starting training')
            timestamp = time()
            for data_batch in data_generator:
                (inputs_v, inputs_length_v, targets_v) = data_batch
                summary_v, global_step_v, result_v, _ = sess.run(
                    fetches=[summary, global_step, result, train_op],
                    feed_dict={inputs: inputs_v, inputs_length: inputs_length_v, targets: targets_v})
                summary_writer.add_summary(summary=summary_v, global_step=global_step_v)

                # Reporting
                if global_step_v % report_period == 0:
                    elapsed = time() - timestamp
                    print("-" * 60)
                    print("Iter %d, Epoch %.0f, Time per iter %.2fs" % (global_step_v, data_generator.progress, elapsed / report_period))
                    print("  Input: %s" % " ".join([embeddings.inv_dictionary[ele] for ele in inputs_v[0, :inputs_length_v[0]]]))
                    print("  Output: %s" % " ".join(["%d:%s" % (idx+1, embeddings.inv_dictionary[ele]) for idx, ele in enumerate(result_v.indices[0, :])]))
                    print("  Target: %s" % embeddings.inv_dictionary[targets_v[0]])
                    print("-" * 60)
                    print("-" * 60)
                    print("-" * 60)
                    timestamp = time()

                # Checkpointing
                if global_step_v % 1000 == 0:
                    save_path = os.path.join(output_path, 'model.ckpt')
                    real_save_path = saver.save(sess=sess, save_path=save_path,
                                                global_step=global_step_v)
                    logging.info('Saved the checkpoint to: {}'
                                 .format(real_save_path))

            # Evaluation
            evaluation_test = VRDEvaluation.from_directory(evaluation_path, embeddings, include_test=True)
            if len(evaluation_test) > 0:
                top1_prec = 0
                top5_prec = 0
                top10_prec = 0
                total_predictions = 0
                outputs = 0
                elapsed = time()
                print("Evaluation samples")
                for eval_batch in EvaluationDataGenerator(evaluation_test, batch_size=batch_size, max_epoch=1):
                    (ev_inputs, ev_inputs_length, ev_targets) = eval_batch
                    eval_results = sess.run(result, feed_dict={inputs: ev_inputs, inputs_length: ev_inputs_length})
                    eval_results = eval_results.indices
                    for i in range(eval_results.shape[0]):
                        if eval_results[i, 0] == ev_targets[i]:
                            top1_prec += 1
                        if ev_targets[i] in eval_results[i, :5]:
                            top5_prec += 1
                        if ev_targets[i] in eval_results[i, :10]:
                            top10_prec += 1
                        total_predictions += 1
                        if random() < 0.3 and outputs < 5:
                            outputs += 1
                            print("  Label: %s. Predicted: %s" % (
                                embeddings.inv_dictionary[ev_targets[i]],
                                " ".join(["%d:%s" % (idx + 1, embeddings.inv_dictionary[ele]) for idx, ele in
                                          enumerate(eval_results[i, :])])))
                            print("  Input: %s" % " ".join(
                                [embeddings.inv_dictionary[ele] for ele in ev_inputs[i, :ev_inputs_length[i]]]))

                print("-" * 60)
                elapsed = time() - elapsed
                print("Evaluation size: %d, time: %.2fs" % (total_predictions, elapsed))
                print("  Precision @  1: %3.2f%%" % ((float(top1_prec) / total_predictions) * 100))
                print("  Precision @  5: %3.2f%%" % ((float(top5_prec) / total_predictions) * 100))
                print("  Precision @ 10: %3.2f%%" % ((float(top10_prec) / total_predictions) * 100))


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
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--sentence-dim', type=int, default=300,
                        help='The dimension of a sentence representation')
    parser.add_argument('--evaluation', required=False,
                        help='Path to the evaluation file')
    parser.add_argument('--report', type=int, default=100, required=False,
                        help='Report every n iterations')
    parser.add_argument('--eval', default=False, action='store_true',
                        help='Evaluate instead of training')

    args = parser.parse_args()
    main()
