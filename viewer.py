import tensorflow as tf
import argparse
from util.rsdae_data_generator import RSDAEDataGenerator


def load_dict(filename):
    print("Loading dict", filename)
    forward = dict()
    reverse = dict()
    with open(filename, 'rb') as reader:
        for line in reader:
            line = line.decode("utf-8").strip()
            idx = len(forward)
            forward[line] = idx
            reverse[idx] = line

    return forward, reverse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gather psn profiles')
    parser.add_argument('--input', required=True, help='Input file', metavar='#')
    parser.add_argument('--words', required=True, help='Word dict', metavar='#')
    parser.add_argument('--postags', required=True, help='POS tag dict', metavar='#')

    args = parser.parse_args()

    print("Initialized with settings:")
    print(" ", vars(args))

    _, reverse_words = load_dict(args.words)
    _, reverse_postags = load_dict(args.postags)

    print("Initialising data generator...")
    data_tensors = RSDAEDataGenerator.get_data_tensor(args.input)
    iterator = data_tensors.make_one_shot_iterator()
    cur_words, cur_postags = iterator.get_next()
    print("Done")

    with tf.Session() as sess:
        print(cur_words.eval())