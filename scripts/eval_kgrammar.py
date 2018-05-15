import os
import argparse
import logging
import pandas as pd
import torch

from check_correct import correct

import seq2seq
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3



parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_folder', help='Give the path to the folder containing test files')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

opt = parser.parse_args()

test_files = os.listdir(opt.test_folder)

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

#################################################################################
# Generate predictor
predictor = Predictor(seq2seq, input_vocab, output_vocab)
#seq_acc = {}
for tf in test_files:
    data_arr = pd.read_csv(os.path.join(opt.test_folder, tf), delimiter='\t', header=None).values
    count1 = 0
    count2 = 0
    for i in range(data_arr.shape[0]):
        src = data_arr[i,0].strip().split()
        tgt = predictor.predict(src)
        tgt1 = tgt[:tgt.index('<eos>')] #without <eos>
        flag2 =False
        if(len(tgt)>=3*len(src)):
            tgt2 =  tgt[:3*len(src)]  #with <eos>
            flag2 = correct(data_arr[i, 0], ' '.join(map(str,tgt1))) #with eos
        flag1 = correct(data_arr[i,0], ' '.join(map(str,tgt1))) #without eos
        if (flag1==True):
            count1 += 1
        if(flag2 == True):
            count2 +=1
    sa1 = count1/(data_arr.shape[0]) #without eos
    sa2 = count2 / (data_arr.shape[0]) #with eos
    print("Sequence Accuracy for {}: without <eos> = {} and with <eos> = {}".format(tf, sa1, sa2))
    #seq_acc[tf] = (sa1, sa2)


# while True:
#         seq_str = raw_input("Type in a source sequence:")
#         seq = seq_str.strip().split()
#         print(predictor.predict(seq))
