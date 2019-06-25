import numpy as np
import random
import string
import argparse
import os
from scripts.produce import get_data

mfolder = '../CommaiMini-^$'
parser = argparse.ArgumentParser()
parser.add_argument('--max_train_com', type=int, help= 'max length of compositions in train', default=4)
parser.add_argument('--max_test_com', type=int, help= 'max length of compositions in test', default=7)
parser.add_argument('--long', type=int, help= 'number of samples in big dataset', default=120000)
parser.add_argument('--short', type=int, help= 'number of samples in small dataset', default=24000)

opt = parser.parse_args()

alphabets = [string.printable[i] for i in range(len(string.printable)-6)]
# for i, word in enumerate(alphabets):
#     print(i, word)
# print(len(alphabets))
# input()
bin_opts = ['yes', 'no']
pidx = int(len(alphabets)/2)
random.shuffle(alphabets)
subset1 = alphabets[:pidx]
subset2 = alphabets[pidx:]
operators = ['copy','and', 'or', 'not']
ponder = 'erm'
eois = '<eos>'
def attn_list(sep):
    ## Get attention
    sep = sep.split(' ')
    temp_attn = []
    indices = [i for i, x in enumerate(sep) if x == "and" or x == "or"]
    temp_attn.extend(indices)
    ver_idx = sep.index('verify')
    temp_attn.extend([ver_idx - 1, ver_idx])
    temp_attn.append(len(sep) - 1)
    return(temp_attn)


def lookup_str(ps, ipt, binary, token, gate=''):
    sep = [token] #['<sep>']
    temp_ipt = list(set(ipt) - set(['and', 'or', 'not'])) #'verify',
    size = random.sample(np.arange(0, opt.max_test_com + 1, dtype=int).tolist(), 1)
    ps.extend(np.random.choice(alphabets, size=size, replace=False).tolist())
    random.shuffle(ps)
    if(binary=='no' and gate != 'not'):
        for word in temp_ipt:
            if word in ps:
                ps.pop(ps.index(word))
        if(len(ps)==0):
            temp_alphabets = list(set(alphabets) - set(temp_ipt))
            ps.append(np.random.choice(temp_alphabets, size=1)[0])
    elif(binary=='no' and gate == 'not'):
        for word in temp_ipt:
            if(ipt[ipt.index(word)-1]=='not'):
                ps.append(word)
            elif(ipt[ipt.index(word)-1] != 'not'):
                ps.pop(ps.index(word))
    sep.extend(ps)
    return (sep)


def copy_gate(ps, token, binary):
    temp_ipt = [] #[token]
    for s in ps:
        temp_ipt.append(s)
    ps = lookup_str(ps, temp_ipt, binary, token) #
    return (ps, temp_ipt)

def and_gate(ps, token,  binary):
    temp_ipt = [] #[token]
    for i, s in enumerate(ps):
        temp_ipt.append(s)
        if (i != len(ps) - 1):
            temp_ipt.append('and')
    ps = lookup_str(ps, temp_ipt, binary, token) #, temp_attn

    return(ps, temp_ipt)

def or_gate(ps, token, binary):
    temp_ipt = [] #[token]
    for i, s in enumerate(ps):
        temp_ipt.append(s)
        if (i != len(ps) - 1):
            temp_ipt.append('or')
    size = random.sample(np.arange(1, len(ps) + 1, dtype=int).tolist(), 1)
    out_str = np.random.choice(ps, size=size, replace=False).tolist()
    out_str = lookup_str(out_str, temp_ipt, binary, token) #, temp_attn
    return(out_str, temp_ipt)

def not_gate(ps, token, binary):
    temp_ipt = [] #[token]
    temp_opt = []
    num_nots = random.sample(np.arange(1,len(ps)+1, dtype=int).tolist(),1)[0]
    not_pfx = random.sample(ps, num_nots)
    cvocab = list(set(alphabets) - set(not_pfx))
    # for pf in not_pfx:
    #     temp_alpha = list(set(alphabets) - set([pf]))
    #     cvocab.append(temp_alpha)
    for i, s in enumerate(ps):
        if (s in not_pfx):
            temp_ipt.append('not')
            temp_ipt.append(s)
            if (i != len(ps) - 1):
                temp_ipt.append('and')
            temp_opt.append(random.sample(cvocab, 1)[0])
        else:
            temp_ipt.append(s)
            temp_opt.append(s)
            if (i != len(ps) - 1):
                temp_ipt.append('and')
    temp_opt = lookup_str(temp_opt, temp_ipt,binary,token, 'not') #, temp_attn
    return (temp_opt, temp_ipt)


def io_strings(word, all_words, comp_len, token):
    ipt = []
    attn = []
    fout = []
    comps = np.random.choice(comp_len, size=len(operators))
    operations = np.random.choice(operators, size=len(operators), replace=False).tolist()
    #random.shuffle(operations)
    for b in bin_opts:

        for i in range(len(comps)):
            tin = []
            tout = []
            ps = [word]
            if (operations[i] == 'copy'):
                str_tup = copy_gate(ps, token, b)
            elif (operations[i] == 'and'):
                ps.extend(random.sample(all_words, comps[i] - 1))
                str_tup = and_gate(ps, token, b)
            elif(operations[i] == 'or'):
                ps.extend(random.sample(all_words, comps[i] - 1))
                str_tup = or_gate(ps, token, b)
            else:
                ps.extend(random.sample(all_words, comps[i] - 1))
                str_tup = not_gate(ps, token, b)
            tout.append(' '.join(map(str, str_tup[0])))
            tin.append(' '.join(map(str, str_tup[1])))
            tin.extend(tout)
            tattn = attn_list(' '.join(map(str, tin)))
            tin.append(eois)
            ipt.append(' '.join(map(str, tin)))
            pre_out = []
            for i in range(len(tattn)-1):
                pre_out.append(ponder)
            pre_out.append(b)
            fout.append(' '.join(map(str, pre_out)))
            tattn.append(tattn[-1] + 1)
            attn.append(' '.join(map(str, tattn)))
    return (ipt, fout, attn)

def train(words, dsize):
    comp_lens = np.arange(2, opt.max_train_com+1, dtype=int).tolist()
    data = np.zeros((dsize, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words)
            for w in words:
                tup = io_strings(w, words, comp_lens, 'verify')
                data[idx:idx+len(tup[0]),0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass
    return data

def unseen(words1, words2, dsize):
    comp_lens = np.arange(2, opt.max_train_com+1, dtype=int).tolist()
    data = np.zeros((dsize, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words1)
            for w in words1:
                tup = io_strings(w, words2, comp_lens, 'verify')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
            random.shuffle(words2)
            for w in words2:
                tup = io_strings(w, words1, comp_lens, 'verify')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass

    return data

def longer(words, dsize):
    comp_lens = np.arange(opt.max_train_com+1, opt.max_test_com+1, dtype=int).tolist()
    data = np.zeros((dsize, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words)
            for w in words:
                tup = io_strings(w, words, comp_lens, 'verify')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass
    return data

def long_unseen(words1, words2, dsize):
    comp_lens = np.arange(opt.max_train_com + 1, opt.max_test_com + 1, dtype=int).tolist()
    data = np.zeros((dsize, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words1)
            for w in words1:
                tup = io_strings(w, words2, comp_lens, 'verify')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()

            random.shuffle(words2)
            for w in words2:
                tup = io_strings(w, words1, comp_lens, 'verify')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass

    return data

data_splits = [os.path.join(mfolder,'Long'), os.path.join(mfolder,'Short')]
data_sizes = [int(opt.long/2), int(opt.short/2)]

for k, size in enumerate(data_sizes):
    train_data, unseen_long_test, unseen_test, longer_test = get_data(size, subset1, subset2)
    print('finished generating produce data, starting generation of verify data')
    tr1 = train(subset1, int(size/2))
    tr2 = train(subset2, int(size/2))
    verify_train = np.vstack((tr1,tr2))
    verify_unseen = unseen(subset1, subset2, int(size/10))
    lg1 = longer(subset1, int(size/20))
    lg2 = longer(subset2, int(size/20))
    verify_longer = np.vstack((lg1, lg2))
    verify_ul = long_unseen(subset1, subset2, int(size/10))

    train_final = np.vstack((verify_train, train_data))
    np.random.shuffle(train_final)
    unseen_final = np.vstack((verify_unseen, unseen_test))
    np.random.shuffle(unseen_final)
    longer_final = np.vstack((verify_longer, longer_test))
    np.random.shuffle(longer_final)
    ul_final = np.vstack((verify_ul, unseen_long_test))
    np.random.shuffle(ul_final)

    num_validation = int(0.1 * train_final.shape[0])
    valid_samples = np.random.choice(np.arange(0, train_final.shape[0]), size=num_validation, replace=False)

    valid_final = train_final[valid_samples]
    train_final = np.delete(train_final, valid_samples, axis=0)

    fnames = ['train', 'validation', 'unseen', 'longer', 'unseen_longer']
    dataset = [train_final, valid_final, unseen_final, longer_final, ul_final]

    for j, fname in enumerate(fnames):
        out_file = open(os.path.join(data_splits[k], 'Verify_Produce_{}.tsv'.format(fname)), 'w')
        file_data = dataset[j]
        for i in range(file_data.shape[0]):
            out_file.write("{}\t{}\t{}\n".format(file_data[i,0], file_data[i,1], file_data[i,2]))

    print('finished generating verify data')