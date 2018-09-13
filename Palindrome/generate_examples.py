import os
import math
import numpy as np
import argparse


def main(args):

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    train_path = os.path.join(args.output_folder,
                              'train_{}.tsv'.format(args.length))
    generate_and_save_examples(args.length, args.num_train_examples,
                               train_path)

    test_path = os.path.join(args.output_folder,
                             'test_{}.tsv'.format(args.length))
    generate_and_save_examples(args.length, args.num_test_examples,
                               test_path)

    if args.longer:
        if args.length_longer < args.length:
            raise ValueError(("Expected longer length to be greater than " +
                              "{}, but got {}.")
                             .format(args.length, args.length_longer))
        longer_path = os.path.join(args.output_folder,
                                   'test_{}.tsv'.format(args.length))
        generate_and_save_examples(args.length_longer,
                                   args.num_longer_examples, longer_path)


def generate_and_save_examples(length, num_examples, file_path):

    if os.path.exists(file_path):
        os.remove(file_path)

    # generate and save training examples
    for _ in range(num_examples):
        X, y = generate_palindrome_example(length)
        with open(file_path, 'a') as file:
            file.write(' '.join(map(str, X)) +
                       '\t' +
                       ' '.join(map(str, y)) +
                       '\n')


def generate_palindrome_example(length, pad_token='<pad>'):
    # Generates a single, random palindrome number of 'length' digits.
    left = [np.random.randint(0, 10) for _ in range(math.ceil(length/2))]
    left = np.asarray(left, dtype=np.int32)
    right = np.flip(left, 0) if length % 2 == 0 else np.flip(left[:-1], 0)
    left, right = left.tolist(), right.tolist()
    palindrome = left + right
    X = palindrome[:-1]
    y = [pad_token] * (len(X)-1) + [palindrome[-1]]

    return X, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', type=str, default='sample1')

    parser.add_argument('-l', '--length', type=int, default=10)
    parser.add_argument('-n-train', '--num-train-examples', type=int,
                        default=10000)
    parser.add_argument('-n-test', '--num-test-examples', type=int,
                        default=1000)
    parser.add_argument('--longer', action='store_true')
    parser.add_argument('-n-longer', '--num-longer-examples', type=int,
                        default=1000)
    parser.add_argument('-l-longer', '--length-longer', type=int,
                        default=15)

    args = parser.parse_args()

    main(args)
