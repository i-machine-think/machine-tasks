# Palindrome Task

The task is to predict the last digit of a palindrome, given all other digits. Therefore, the network has to keep in memory the first digit it has seen.

## Examples

| Length| Input       |Output|
|-------|-------------|------|
|10     | `012344321` | `0`  |
|11     | `0123454321`| `0`  |

## Dataset generation
A new training and test set can be generated with [`generate_examples.py`](generate_examples.py). With e.g. `-l 10` a training and testing file with palindrome lengths of 10 is generated.  
Note that the output in the file is padded, in order to work smoothly with the machine library.
