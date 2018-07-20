import numpy as np
import os
import matplotlib.pyplot as plt

mfolder = '../CommaiMini-^$/New_AG'
data_split = os.path.join(mfolder,'Long')


def ops_count(data, name):
    or_op = []
    and_op = []
    not_op = []
    copy = []

    for d in data:
        if ('or' in d):
            or_op.append(d)
        elif('not' in d):
            not_op.append(d)
        elif('and' in d):
            and_op.append(d)
        else:
            copy.append(d)

    fig, ax = plt.subplots()
    p1 = ax.bar([1,2,3, 4], [len(copy), len(or_op), len(and_op), len(not_op)])
    ax.set_xticks([1,2,3, 4])
    ax.set_xticklabels(('copy','or', 'and', 'not'))
    ax.set_ylabel('Instances per operator')
    ax.set_xlabel('Operators')
    ax.set_title('{} case'.format(name))

    for p in p1:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., 0.90*height,
                '%d' % int(height),
                ha='center', va='bottom')
    #plt.savefig(os.path.join(mfolder, 'Stats', '{}-eps.eps'.format(name)), format='eps')
    plt.show()

def file_stats(fname):
    in_file = open(os.path.join(data_split, 'Verify_Produce_{}.tsv'.format(fname)), 'r')
    all_lines = in_file.readlines()
    data_arr = np.zeros((len(all_lines), 3), dtype=object)
    err_chk = np.zeros((len(all_lines), 3), dtype=object)
    for idx, line in enumerate(all_lines):
        targets = line.strip('\n').split('\t')
        for i, tgt in enumerate(targets):
            data_arr[idx, i] = tgt.split(' ')
            err_chk[idx, i] = tgt

    verify = []
    produce = []
    for i in range(data_arr.shape[0]):
        if('verify' in data_arr[i,0]):
            verify.append(data_arr[i,0])
        else:
            produce.append(data_arr[i,0])

    ops_count(verify, '{}_verify'.format(fname))
    ops_count(produce, '{}_produce'.format(fname))

    return err_chk

fnames = ['train', 'validation', 'unseen', 'longer', 'unseen_longer']
operators = ['and', 'or', 'not']

err_report = []
for fname in fnames:
    err_arr = file_stats(fname)
    err = []
    for i in range(err_arr.shape[0]):
        if('verify' in err_arr[i,0].split(' ')):
            temp_str = err_arr[i,0].split('verify')[0]
        else:
            temp_str = err_arr[i,0].split('produce')[0]

        if (temp_str[-1] in operators or temp_str[-1]==''):
            err.append(err_arr[i,0])
    err_report.append(err)

print(err_report)






