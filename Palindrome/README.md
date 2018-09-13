# Palindrome Task

The task is to predict the last digit of a palindrome, given all other digits. Therefore, the network has to keep in memory the first digit it has seen.

## Examples

| Length| Input       |Output|
|-------|-------------|------|
|10     | `012344321` | `0`  |
|11     | `0123454321`| `0`  |

## Dataset Generation
A new training and test set can be generated with [`generate_examples.py`](generate_examples.py). With e.g. `-l 10` a training and testing file with palindrome lengths of 10 is generated.  

*Note*: The output in the file is padded, in order to work smoothly with the machine library.  

The samples in the [`sample1`](sample1/) folder are generated with the following command:
`python3 generate_examples.py --output-folder sample1 -s 1 -l 5 8 10 12 15 20 25 30 40 50 --valid --test`  

*Note*: A seed should be provided with `-s` when generating a new sample, so the generated data is different from previous examples. Check the `args.txt` file in the sample folder for the used seed.


### Test Set with longer Examples
Additionally, a function is provided to generate a test set with longer examples. Therefore, simply add `--longer` and `--length-longer 100` to the command. A test set named `test_longer_100.tsv` will then be created.
