import os
import math
import numpy as np
import argparse


def main(args):

    print(args)

    np.random.seed(args.seed)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for length in args.length:
        train_path = os.path.join(args.output_folder,
                                  'train_{:03d}.tsv'.format(length))
        generate_and_write_examples(length, args.num_train_examples,
                                    train_path)

        if args.valid:
            valid_path = os.path.join(args.output_folder,
                                      'valid_{:03d}.tsv'.format(length))
            generate_and_write_examples(length, args.num_eval_examples,
                                        valid_path)

        if args.test:
            test_path = os.path.join(args.output_folder,
                                     'test_{:03d}.tsv'.format(length))
            generate_and_write_examples(length, args.num_eval_examples,
                                        test_path)

    if args.longer:
        if args.length_longer < max(args.length):
            raise ValueError(("Expected longer length to be greater than " +
                              "{}, but got {}.")
                             .format(max(args.length), args.length_longer))
        longer_path = os.path.join(args.output_folder, 'test_longer_' +
                                   '{:03d}.tsv'.format(args.length_longer))
        generate_and_write_examples(args.length_longer,
                                    args.num_eval_examples, longer_path)

    with open(os.path.join(args.output_folder, 'args.txt'), 'w') as file:
        file.write(str(args))


def generate_and_write_examples(length, num_examples, file_path):

    if os.path.exists(file_path):
        os.remove(file_path)

    # generate and save training examples
    for _ in range(num_examples):
        X, y = generate_palindrome_example(length)
        with open(file_path, 'a') as file:
            file.write(' '.join(map(str, X)) + '\t' +
                       ' '.join(map(str, y)) + '\n')


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
    parser.add_argument('-s', '--seed', type=int, default=1)

    parser.add_argument('-l', '--length', nargs='*', type=int, default=[10])
    parser.add_argument('-n-train', '--num-train-examples', type=int,
                        default=10000)

    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-n-eval', '--num-eval-examples', type=int,
                        default=1000,
                        help='Number of examples in evaluation sets.')

    parser.add_argument('--longer', action='store_true')
    parser.add_argument('-l-longer', '--length-longer', type=int,
                        default=15)

    args = parser.parse_args()

    main(args)
