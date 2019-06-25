import numpy as np
import os

mfolder = '../CommaiMini-^$/New_AG'
data_split = os.path.join(mfolder,'Long')

def write_file(op_data, name, op_name):
    data_arr = np.zeros((len(op_data),3), dtype=object)
    for i in range(len(op_data)):
        for j in range(data_arr.shape[1]):
            data_arr[i,j] = ' '.join(map(str,op_data[i][j]))

    out_file = open(os.path.join(mfolder,name.split('_')[0], name+'_{}.tsv'.format(op_name)), 'w')
    for i in range(data_arr.shape[0]):
        out_file.write("{}\t{}\t{}\n".format(data_arr[i, 0], data_arr[i, 1], data_arr[i, 2])) #\t{}


def ops_count(data, name):
    or_op = []
    and_op = []
    not_op = []
    copy = []

    for d in data:
        if ('or' in d[0]):
            or_op.append(d)
        elif('not' in d[0]):
            not_op.append(d)
        elif('and' in d[0]):
            and_op.append(d)
        else:
            copy.append(d)

    write_file(or_op, name, 'or')
    write_file(not_op, name, 'not')
    write_file(and_op, name, 'and')
    if(len(copy) != 0):
        write_file(copy, name, 'copy')


def file_stats(fname):
    in_file = open(os.path.join(data_split, 'Verify_Produce_{}.tsv'.format(fname)), 'r')
    all_lines = in_file.readlines()
    data_arr = np.zeros((len(all_lines), 3), dtype=object)
    for idx, line in enumerate(all_lines):
        targets = line.strip('\n').split('\t')
        for i, tgt in enumerate(targets):
            data_arr[idx, i] = tgt.split(' ')

    verify = []
    produce = []
    for i in range(data_arr.shape[0]):
        if('verify' in data_arr[i,0]):
            verify.append(data_arr[i])
        else:
            produce.append(data_arr[i])

    ops_count(verify, 'verify_{}'.format(fname))
    ops_count(produce, 'produce_{}'.format(fname))

    return (verify, produce)

fnames = ['unseen', 'longer', 'unseen_longer']

for fname in fnames:
    file_stats(fname)






