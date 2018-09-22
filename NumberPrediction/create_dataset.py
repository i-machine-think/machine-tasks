import os
import csv
import wget
import tarfile
import argparse


def main(args):

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    file_path = os.path.join(args.data_dir, 'rnn_agr_simple.tar.gz')
    if not os.path.exists(file_path):
        wget.download(args.download_url, out=file_path)
        print()

    # Extract files
    with tarfile.open(file_path, 'r:gz', ignore_zeros=True) as tar:
        for member in tar.getmembers():

            # skip dummy files
            if os.path.split(member.name)[-1][:1] == '.' or \
             os.path.split(member.name)[-1] == 'README.txt':
                continue

            tar.extract(member, path=args.data_dir)

    # preprocess
    extract_dir = os.path.join(args.data_dir, 'rnn_agr_simple')
    for file in os.listdir(extract_dir):

        with open(os.path.join(extract_dir, file), 'r') as in_file, \
         open(os.path.join(args.data_dir, file), 'w') as ot_file:
            reader = csv.reader(in_file, delimiter='\t')
            writer = csv.writer(ot_file, delimiter='\t')

            for r in reader:
                assert len(r) == 2 and r[0] in ['VBZ', 'VBP'], \
                    "Something's wrong with this line:\n{}".format(r)

                input = r[1]
                # add <pad> tokens to target to make it compatible with machine
                target = ['<pad>'] * (len(input.split())-1) + [r[0]]
                target = ' '.join(target)

                writer.writerow([input, target])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--download_url', type=str, default='http://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    main(args)
