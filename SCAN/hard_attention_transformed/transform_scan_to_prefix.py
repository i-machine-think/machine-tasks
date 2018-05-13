import os
from os import listdir
from os.path import isfile, join

in_dir = os.path.join("machine-tasks", "SCAN", "hard_attention", "length_split")
out_dir = os.path.join("machine-tasks", "SCAN", "hard_attention_transformed", "length_split")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

in_filenames = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

def transform(split):

    if "and" in split:
        index = split.index("and")
        split.pop(index)
        split = ["and"] + transform(split[:index]) + transform(split[index:])
        return split

    if "after" in split:
        index = split.index("after")
        split.pop(index)
        split = ["after"] + transform(split[:index]) + transform(split[index:])
        return split

    if "twice" in split:
        index = split.index("twice")
        split.pop(index)
        split = ["twice"] + transform(split)
        return split

    if "thrice" in split:
        index = split.index("thrice")
        split.pop(index)
        split = ["thrice"] + transform(split)
        return split

    if "around" in split:
        index = split.index("around")
        split.pop(index)
        split = ["around"] + transform(split)
        return split

    if "opposite" in split:
        index = split.index("opposite")
        split.pop(index)
        split = ["opposite"] + transform(split)
        return split

    if "left" in split:
        index = split.index("left")
        split.pop(index)
        split = ["left"] + transform(split)
        return split

    if "right" in split:
        index = split.index("right")
        split.pop(index)
        split = ["right"] + transform(split)
        return split

    return split

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

        transformed_input_seq = transform(input_seq.split(" "))
        transformed_input_seq = " ".join(transformed_input_seq)

        out_file.write("{}\t{}\t{}\n".format(transformed_input_seq, output_seq, attn_seq))

    in_file.close()
    out_file.close()