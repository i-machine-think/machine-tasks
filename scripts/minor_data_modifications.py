# This script can be used to either add or remove input EOS symbols, and add attentive guidance indices
# Please extend this script with extra functionality (and for new data sets) if necessary

import os

# The directory containing the input .tsv or .csv files. Make sure you call the script from the correct directory
input_base_directory = os.path.join("..", "LookupTables")
# Boolean to indicate whether we should also process all files in subdirectories of 'input_base_directory'
full_traversal = True
# The directory in which all the processed files will be stored as .tsv files
output_base_directory = os.path.join('..', 'test_output')

ADD_INPUT_EOS = False
REMOVE_INPUT_EOS = False
INPUT_EOS_SYMBOL = '.'

USE_OUTPUT_EOS = True

ADD_ATTN = 'lookup' # Set to None to not add attention, or set to 'lookup' to add diagonal attention

if ADD_ATTN:
    if ADD_ATTN == 'lookup':
        # This method can be altered to either let the output EOS attend to the input EOS or to something else
        def add_lookup_attention(input_seq, output_seq):
            if USE_OUTPUT_EOS:
                attn_seq = [str(index) for index in list(range(len(output_seq)+1))]
            else:
                attn_seq = [str(index) for index in list(range(len(output_seq)))]

            return attn_seq

        ADD_ATTN = add_lookup_attention

SHOW_EXAMPLES = True

if __name__ == '__main__':
    if not os.path.exists(output_base_directory):
        os.makedirs(output_base_directory)

    # If full_traversal: get generator over all subdirectories
    # Else: Create dummy generator with only 1 object
    if full_traversal:
        directory_generator = os.walk(input_base_directory)
    else:
        dir_path = input_base_directory
        filenames = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f))]
        directory_generator = (dir_path, [], filenames)

    for input_directory, dirnames, filenames in directory_generator:
        # Create output directory
        relative_path = os.path.relpath(input_directory, input_base_directory)
        output_directory = os.path.join(output_base_directory, relative_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # loop over tsv files
        for input_filename in filenames:
            print("Processing file {}".format(os.path.join(input_directory, input_filename)))

            if not input_filename.endswith('.tsv') and not input_filename.endswith('.csv'):
                continue
            output_filename = input_filename.replace('.csv', '.tsv')

            input_file = open(os.path.join(input_directory, input_filename), 'r')
            output_file = open(os.path.join(output_directory, output_filename), 'w')

            show_example = True
            for line in input_file:
                line_split = line.strip().split('\t')

                if len(line_split) == 2:
                    input_seq = line_split[0].split()
                    output_seq = line_split[1].split()
                    attn_seq = None
                else:
                    input_seq = line_split[0].split()
                    output_seq = line_split[1].split()
                    attn_seq = line_split[2].split()

                if ADD_INPUT_EOS:
                    input_seq.append(INPUT_EOS_SYMBOL)
                if REMOVE_INPUT_EOS:
                    input_seq = input_seq[:-1]
                if ADD_ATTN is not None:
                    attn_seq = ADD_ATTN(input_seq, output_seq)

                # Check whether output is valid
                if attn_seq:
                    if USE_OUTPUT_EOS:
                        assert len(output_seq)+1 == len(attn_seq), "Attention sequence should be exactly 1 item longer than the output sequence"
                    else:
                        assert len(output_seq) == len(attn_seq), "Output sequence and attention sequence should be equally long"

                    max_attn = max(int(attn) for attn in attn_seq)
                    assert max_attn <= len(input_seq), "Can't attend to non-existent input symbol"

                # Compose output string
                input_seq = " ".join(input_seq)
                output_seq= " ".join(output_seq)
                output_string = "{}\t{}".format(input_seq, output_seq)
                if attn_seq:
                    attn_seq = " ".join(attn_seq)
                    output_string += "\t{}".format(attn_seq)
                output_string += "\n"

                output_file.write(output_string)

                if show_example and SHOW_EXAMPLES:
                    print("Example  in: {}Example out: {}".format(line, output_string))
                    show_example = False

            input_file.close()
            output_file.close()


    # Remove all created directories that are empty. Apparantly, these did not have .tsv files in them.
    for output_directory, dirnames, filenames in os.walk(output_base_directory):
        try:
            os.rmdir(output_directory)
        except OSError:
            pass