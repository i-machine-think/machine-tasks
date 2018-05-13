import os
from os import listdir
from os.path import isfile, join

in_dir = os.path.join("..", "hard_attention", "addprim_turn_left_split")
out_dir = os.path.join("addprim_turn_left_split")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

in_filenames = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

def transform(input_split, attn_split, relative_index):

    for conj in ["and", "after"]:
        if conj in input_split:
            index = input_split.index(conj)

            input_split.pop(index)

            # Since we move AND or AFTER to the front, we must add 1 to every attention index that was somehwere between the start of the sentence (0) and the index of AND or AFTER
            attn_split = [attn + 1 if attn < index else attn for attn in attn_split]

            transformed_input_seq_1, attn_split = transform(input_split[:index], attn_split, 1)
            transformed_input_seq_2, attn_split = transform(input_split[index:], attn_split, index + 1)

            input_split = [conj] + transformed_input_seq_1 + transformed_input_seq_2
            
            return input_split, attn_split

    ordered_application = ['thrice', 'twice'] + ['around', 'opposite'] + ['left', 'right']

    for word in ordered_application:
        if word in input_split:
            index = input_split.index(word)

            input_split.pop(index)

            # Update attention indices
            for i in range(len(attn_split)):
                # relative_index + index is the absolute index in the sequence.
                # Since we move this word to the front (of the subsequence), we must update all pointers
                # to it in the attention list to 0 (+relative_index)
                if attn_split[i] == relative_index + index:
                    attn_split[i] = relative_index

                else:
                    # Else, since we move that word to the start of the sequence, we must update all attention indices
                    # that were pointing to words that lie between the start of the (sub)squence and the index of the word
                    # we move to the front. We must also check whether the word was not already at the front. (actually I don't think this can happen in SCAN)
                    if attn_split[i] >= relative_index and attn_split[i] < relative_index + index and index != 0:
                        attn_split[i] += 1

            transformed_input_seq, attn_split = transform(input_split, attn_split, relative_index + 1)

            input_split = [word] + transformed_input_seq

            return input_split, attn_split

    if len(input_split) > 1 or input_split[0] not in ['jump', 'run', 'walk', 'look', 'turn']:
        raise("Unkown word in sequence")

    return input_split, attn_split

for in_filename in in_filenames:
    out_filename = in_filename

    if out_filename.endswith('.csv') or out_filename.endswith('.txt'):
        out_filename = out_filename[:-4] + '.tsv'

    in_file = open(os.path.join(in_dir, in_filename), 'r')
    out_file = open(os.path.join(out_dir, out_filename), 'w')

    for line in in_file:
        line_split = line.strip().split('\t')
        input_seq = line_split[0]
        output_seq = line_split[1]
        attn_seq = line_split[2]

        transformed_input_seq, transformed_attn_seq = transform(input_seq.split(" "), [int(attn) for attn in attn_seq.split(" ")], 0)
        transformed_input_seq = " ".join(transformed_input_seq)
        transformed_attn_seq = " ".join([str(attn) for attn in transformed_attn_seq])

        out_file.write("{}\t{}\t{}\n".format(transformed_input_seq, output_seq, transformed_attn_seq))

    in_file.close()
    out_file.close()