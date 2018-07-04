import numpy as np
import random
import argparse
import string

mfolder = '../CommaiMini-^$'
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--max_train_com', type=int, help= 'max length of compositions in train', default=4)
parser.add_argument('--max_test_com', type=int, help= 'max length of compositions in test', default=7)


opt = parser.parse_args()

alphabets = [string.printable[i] for i in range(len(string.printable)-6)]
pidx = int(len(alphabets)/2)
random.shuffle(alphabets)
subset1 = alphabets[:pidx]
subset2 = alphabets[pidx:]
operators = ['and', 'or', 'not']
ponder = 'erm'
eois = '<eos>'
def attn_list(sep):
    temp_attn = []
    indices = [i for i, x in enumerate(sep) if x == "and" or x== "or"]
    temp_attn.extend(indices)
    ver_idx = sep.index('produce')
    temp_attn.extend([ver_idx - 1, ver_idx])
    return(temp_attn)


def and_gate(ps, token):
    temp_ipt = [] #[token]
    for i, s in enumerate(ps):
        temp_ipt.append(s)
        if (i != len(ps) - 1):
            temp_ipt.append('and')
    random.shuffle(ps)
    temp_ipt.append(token)
    temp_attn = attn_list(temp_ipt)
    temp_ipt.append(eois)
    pre_opt = []
    for i in range(len(temp_attn) - 1):
        pre_opt.append(ponder)
    prod_attn = temp_attn[-1]
    for i in range(len(ps) - 1):
        temp_attn.append(prod_attn)
    pre_opt.extend(ps)
    temp_attn.append(temp_attn[-1] + 1)
    return(pre_opt, temp_ipt, temp_attn)

def or_gate(ps, token):
    temp_ipt = [] #[token]
    for i, s in enumerate(ps):
        temp_ipt.append(s)
        if (i != len(ps) - 1):
            temp_ipt.append('or')
    size = random.sample(np.arange(1, len(ps) + 1, dtype=int).tolist(), 1)
    out_str = np.random.choice(ps, size=size, replace=False).tolist()
    #out_str.append('<eos>')
    temp_ipt.append(token)
    temp_attn = attn_list(temp_ipt)
    temp_ipt.append(eois)
    pre_opt = []
    for i in range(len(temp_attn) - 1):
        pre_opt.append(ponder)
    prod_attn = temp_attn[-1]
    for i in range(len(out_str) - 1):
        temp_attn.append(prod_attn)
    pre_opt.extend(out_str)
    temp_attn.append(temp_attn[-1] + 1)
    return(pre_opt, temp_ipt, temp_attn)

def not_gate(ps, token):
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
            if (i != len(ps) - 1):
                temp_ipt.append('and')
            temp_opt.append(s)
    #temp_opt.append('<eos>')
    temp_ipt.append(token)
    temp_attn = attn_list(temp_ipt)
    temp_ipt.append(eois)
    pre_opt = []
    for i in range(len(temp_attn)-1):
        pre_opt.append(ponder)
    prod_attn = temp_attn[-1]
    for i in range(len(temp_opt)-1):
        temp_attn.append(prod_attn)
    pre_opt.extend(temp_opt)
    temp_attn.append(temp_attn[-1] + 1)
    return (pre_opt, temp_ipt, temp_attn)



def io_strings(word, all_words, comp_len, token):
    ipt = []
    out = []
    attn = []
    comps = np.random.choice(comp_len, size=len(operators))
    operations = np.random.choice(operators, size=len(operators), replace=False).tolist()
    random.shuffle(operations)
    for i in range(len(comps)):
        ps = [word]
        ps.extend(np.random.choice(all_words, size=comps[i]-1).tolist())
        if (operations[i] == 'and'):
            str_tup = and_gate(ps, token)
        elif(operations[i] == 'or'):
            str_tup = or_gate(ps, token)
        else:
            str_tup = not_gate(ps, token)
        out.append(' '.join(map(str, str_tup[0])))
        ipt.append(' '.join(map(str, str_tup[1])))
        attn.append(' '.join(map(str, str_tup[2])))
    return (ipt, out, attn)

def train(words, size):
    comp_lens = np.arange(2, opt.max_train_com+1, dtype=int).tolist()
    data = np.zeros((size, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words)
            for w in words:
                tup = io_strings(w, words, comp_lens, 'produce')
                data[idx:idx+len(tup[0]),0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if(idx > data.shape[0]-len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass

    return data

def unseen(words1, words2, size):
    comp_lens = np.arange(2, opt.max_train_com+1, dtype=int).tolist()
    data = np.zeros((size, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words1)
            for w in words1:
                tup = io_strings(w, words2, comp_lens, 'produce')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()

            random.shuffle(words2)
            for w in words2:
                tup = io_strings(w, words1, comp_lens, 'produce')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass

    return data

def longer(words, size):
    comp_lens = np.arange(opt.max_train_com+1, opt.max_test_com+1, dtype=int).tolist()
    data = np.zeros((size, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words)
            for w in words:
                tup = io_strings(w, words, comp_lens, 'produce')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass
    return data

def long_unseen(words1, words2, size):
    comp_lens = np.arange(opt.max_train_com + 1, opt.max_test_com + 1, dtype=int).tolist()
    data = np.zeros((size, 3), dtype=object)
    idx = 0
    try:
        while idx < data.shape[0]:
            random.shuffle(words1)
            for w in words1:
                tup = io_strings(w, words2, comp_lens, 'produce')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
            random.shuffle(words2)
            for w in words2:
                tup = io_strings(w, words1, comp_lens, 'produce')
                data[idx:idx + len(tup[0]), 0] = tup[0]
                data[idx:idx + len(tup[0]), 1] = tup[1]
                data[idx:idx + len(tup[0]), 2] = tup[2]
                idx += len(tup[0])
                if (idx > data.shape[0] - len(operators)):
                    raise StopIteration()
    except StopIteration:
        pass

    return data

def get_data(num_samples):
    tr1 = train(subset1, int(num_samples/2))
    tr2 = train(subset2, int(num_samples/2))
    train_data = np.vstack((tr1,tr2))
    unseen_test = unseen(subset1, subset2, num_samples)
    lg1 = longer(subset1, int(num_samples/2))
    lg2 = longer(subset2, int(num_samples/2))
    longer_test = np.vstack((lg1, lg2))
    unseen_long_test = long_unseen(subset1, subset2, num_samples)

    return(train_data, unseen_long_test, unseen_test, longer_test)

