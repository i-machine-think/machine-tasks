# -*- coding: utf-8 -*-

"""
Script to generate of the lookup tables problem.

Running this script will save the following files in /dir/ or /dir/sample<i>/ if n_samples > 1:
- train.tsv
- validation.tsv
- heldout_inputs.tsv
- heldout_compositions
- heldout_tables
- new_compositions
- longer_seen_1.tsv
- longer_incremental_1.tsv
- longer_new_1.tsv
...
- longer_seen_n.tsv
- longer_incremental_n.tsv
- longer_new_n.tsv

with n is the max number of additional compositions in test compared to train.

Help : `python make_lookup_tables.py -h`
"""

from __future__ import unicode_literals, division, absolute_import, print_function

import sys
import os
import itertools
import random
import operator
import warnings
import argparse
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


### MAIN ###
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Script to generate of the lookup tables problem.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dir', default='.', help='Path to the directory where to save the generated data.')
    parser.add_argument('-s', '--n-samples', type=int, default=1, help='Number of different samples to generate. If greater than 1, will save the files for a single sample in /dir/sample<i>/*.tsv.')

    parser.add_argument('-v', '--validation-size', type=float, default=0.05, help='Percentage of training set to use as validation.')
    parser.add_argument('-c', '--max-composition-train', type=int, default=4, help='Max length of compositions in training set, also the length of compositions of the test sets.')
    parser.add_argument('-t', '--n-unary-tables', type=int, default=8, help='Number of different lookup tables.')
    parser.add_argument('-T', '--n-heldout-tables', type=int, default=2, help='Number of tables that would only be seen in unary.')
    parser.add_argument('-C', '--n-heldout-compositions', type=int, default=50, help='Number of compositions to randomly remove.')
    parser.add_argument('-I', '--n-heldout-inputs', type=int, default=2, help='Number of inputs to heldout from training.')
    parser.add_argument('-l', '--n-longer', type=int, default=5, help='Number of additional tables to add to `longer` test data.')
    parser.add_argument('--reverse', action='store_true', help='Reverses the input sequence to match the mathematical composition. I.e if given, then uses `t1(t2(input))` without parenthesis instead of `input t2 t1`.')
    parser.add_argument('--not-copy-input', action='store_true', help='Removes the copy of the input in the target sequence.')
    parser.add_argument('--not-intermediate', action='store_true', help='Removes intermediate step in the target sequence.')
    parser.add_argument('--not-shuffle', action='store_true', help='Disables shuffling of the outputed datasets')
    parser.add_argument('--not-stratify', action='store_true', help='Disables the balancing of the lookups in train and validation set.')
    parser.add_argument('--is-target-attention', action='store_true', help='Append the target attention as an additional column.')
    parser.add_argument('-e', '--eos', default='.', help='EOS token to append at the end of each input.')
    parser.add_argument('-b', '--bound-test', type=int, default=5000, help='Bound the number of rows in each test files.')
    parser.add_argument('-a', '--alphabet', metavar=('letter1', 'letter2'), nargs='*', default=['0', '1'], help='Possible characters given as input.')
    parser.add_argument('-r', '--n-repeats', type=int, default=3, help='Number of characters in `alphabet` used in each input and output. If the alphabet==bits, then this corresponds to `n` in `n-bits`.')
    parser.add_argument('-S', '--seed', type=int, default=123, help='Random seed.')

    args = parser.parse_args(args)
    return args


def main(args):
    random.seed(args.seed)
    for sample in range(args.n_samples):
        seed = args.seed if args.n_samples == 1 else random.randint(0, 1e5)
        out = table_lookup_dataset(validation_size=args.validation_size,
                                   max_composition_train=args.max_composition_train,
                                   n_unary_tables=args.n_unary_tables,
                                   n_heldout_tables=args.n_heldout_tables,
                                   n_heldout_compositions=args.n_heldout_compositions,
                                   n_heldout_inputs=args.n_heldout_inputs,
                                   add_composition_test=args.n_longer,
                                   is_reverse=args.reverse,
                                   is_copy_input=not args.not_copy_input,
                                   is_intermediate=not args.not_intermediate,
                                   is_shuffle=not args.not_shuffle,
                                   is_stratify=not args.not_stratify,
                                   is_target_attention=args.is_target_attention,
                                   eos=args.eos,
                                   bound_test=args.bound_test,
                                   seed=seed,
                                   alphabet=args.alphabet,
                                   n_repeats=args.n_repeats)

        names = ("train", "validation", "heldout_inputs", "heldout_compositions", "heldout_tables",
                 "new_compositions", "longer_seen", "longer_incremental", "longer_new")

        for data, name in zip(out, names):
            path = args.dir if args.n_samples == 1 else os.path.join(args.dir, "sample{}".format(sample + 1))
            _save_tsv(data, name, path)


### FUNCTIONS ###
def table_lookup_dataset(validation_size=0.11,
                         max_composition_train=2,
                         n_unary_tables=8,
                         n_heldout_tables=2,
                         n_heldout_compositions=8,
                         n_heldout_inputs=2,
                         add_composition_test=1,
                         is_reverse=False,
                         is_copy_input=True,
                         is_intermediate=True,
                         is_shuffle=True,
                         is_stratify=True,
                         is_target_attention=False,
                         eos=".",
                         bound_test=10**4,
                         seed=123,
                         **kwargs):
    r"""Prepare the table lookup dataset.

    Args:
        validation_size (float, optional): max length of compositions in training set.
        max_composition_train (int, optional): max length of compositions in training set.
        n_unary_tables (int, optional): number of different lookup tables.
        n_heldout_tables (int, optional): number of tables that would only be seen in unary.
        n_heldout_compositions (int, optional): number of compositions of len `max_composition_train`
            to heldout from training.
        n_heldout_inputs (int, optional): the number of inputs of tables of len `max_composition_train`
            to heldout from training.
        add_composition_test (int, optional): additional composition to add for the `longer_*` test data.
            Those test sets will then include compositions between `max_composition_train` and
            `max_composition_train + add_composition_test` tables.
        is_reverse (bool, optional): whether to reverse the  input sequence to match the mathematical composition.
            I.e if given, then uses `t1(t2(input))` without parenthesis instead of `input t2 t1`.'
        is_copy_input (bool, optional): whether to include a copy of the initial input results in the output.
        is_intermediate (bool, optional): whether to include intermediate results in the output.
        is_shuffle (bool, optional): whether to shuffle the outputed datasets.
        is_stratify (bool, optional): whether to split validation to approximately balance each lookup table.
            `validation_size` may have to be larger when using this.
        is_target_attention (bool, optional): whether to append the target attention as an additional column.
        eos (str, optional): token to append at the end of each input.
        bound_test (int, optional): bounds the number of rows in each test files.
        seed (int, optional): sets the seed for generating random numbers.
        kwargs: Additional arguments to `create_N_table_lookup`.

    Returns:
        train (pd.Series): dataframe of all multiary training examples. Contains all the unary functions.
            The index is the input and value is the target.
        validation (pd.Series): dataframe of all multiary examples use for validation.
        heldout_inputs (pd.Series): dataframe of inputs that have not been seen during training but the mapping have.
        heldout_compositions (pd.Series): dataframe of multiary composition that have never been seen during training.
        heldout_tables (pd.Series): dataframe of multiary composition that are made up of one table that has
            never been seen in any multiary composition during training.
        new_compositions (pd.Series): dataframe of multiary composition that are made up of 2 tables that have
            never been seen in any multiary composition during training.
        longer_seens (list of pd.Series): list of len `add_composition_test`. Where the ith element is a dataframe
            composed of `max_composition_train+i` tables that have all been composed in the training set.
        longer_incrementals (list of pd.Series): list of len `add_composition_test`. Where the ith element is a
            dataframe composed of `max_composition_train+i` tables, with at least one that been composed in the
            training set and at least one that hasn't.
        longer_news (list of pd.Series): ist of len `add_composition_test`. Where the ith element is a
            dataframe composed of `max_composition_train+i` tables that have never been composed in the training set.
    """
    assert " " not in eos, "Cannot have spaces in the <eos> token."
    if not is_copy_input and is_target_attention:
        raise NotImplementedError("`is_target_attention` with `is_copy=False` not implemented yet.")

    np.random.seed(seed)
    random.seed(seed)

    unary_functions = create_N_table_lookup(N=n_unary_tables, seed=seed, **kwargs)
    n_inputs = len(unary_functions[0])
    names_unary_train = {t.name for t in unary_functions[:-n_heldout_tables]}
    names_unary_test = {t.name for t in unary_functions[-n_heldout_tables:]}
    multiary_functions = [[reduce(lambda x, y: compose_table_lookups(x, y, is_intermediate=is_intermediate),
                                  fs)
                           for fs in itertools.product(unary_functions, repeat=repeat)]
                          for repeat in range(2, max_composition_train + 1)]
    longest_multiary_functions = multiary_functions[-1]
    multiary_functions = flatten(multiary_functions[:-1])
    longest_multiary_train, heldout_tables, new_compositions = _split_seen_unseen_new(longest_multiary_functions,
                                                                                      names_unary_train,
                                                                                      names_unary_test)
    multiary_train, _, _ = _split_seen_unseen_new(multiary_functions,
                                                  names_unary_train,
                                                  names_unary_test)
    random.shuffle(longest_multiary_train)

    # heldout
    heldout_compositions = longest_multiary_train[-n_heldout_compositions:]

    longest_multiary_train = longest_multiary_train[:-n_heldout_compositions]
    drop_inputs = [np.random.choice(table.index, n_heldout_inputs, replace=False)
                   for table in longest_multiary_train]
    heldout_inputs = [table[held_inputs] for held_inputs, table in zip(drop_inputs, longest_multiary_train)]

    longest_multiary_train = [table.drop(held_inputs) for held_inputs, table in zip(drop_inputs,
                                                                                    longest_multiary_train)]

    # longer
    longer_seens = []
    longer_incrementals = []
    longer_news = []
    longer = [compose_table_lookups(x, y) for x, y in itertools.product(unary_functions, longest_multiary_functions)]
    for _ in range(add_composition_test):
        longer_seen, longer_incremental, longer_new = _split_seen_unseen_new(longer,
                                                                             names_unary_train,
                                                                             names_unary_test)

        for longer_i, longer_i_list in zip([longer_seen, longer_incremental, longer_new],
                                           [longer_seens, longer_incrementals, longer_news]):
            # uses round(bound_test/n_inputs) because at that moment we have a list of composed tables with each
            # `n_inputs` rows. At the end we will merge those and bound_test should filter the total number of rows.
            if len(longer_i) * n_inputs > bound_test:
                warnings.warn("Randomly select tables as len(longer)={} is larger than bound_test={}.".format(n_inputs * len(longer_i),
                                                                                                              bound_test))
                longer_i = random.sample(longer_i, round(bound_test / n_inputs))

            longer_i_list.append(longer_i)

        longer = flatten([longer_seens[-1], longer_incrementals[-1], longer_news[-1]])
        longer = [compose_table_lookups(x, y) for x, y in itertools.product(unary_functions, longer)]

    # formats
    longer_seens = _merge_format_inputs(longer_seens, is_shuffle, bound_test=bound_test, seed=seed,
                                        is_reverse=is_reverse, eos=eos)
    longer_incrementals = _merge_format_inputs(longer_incrementals, is_shuffle, bound_test=bound_test,
                                               seed=seed, is_reverse=is_reverse, eos=eos)
    longer_news = _merge_format_inputs(longer_news, is_shuffle, bound_test=bound_test, seed=seed,
                                       is_reverse=is_reverse, eos=eos)

    multiary_train += longest_multiary_train
    building_blocks = (unary_functions, multiary_train, heldout_inputs, heldout_compositions, heldout_tables, new_compositions)
    # don't bound test because size check after
    building_blocks = _merge_format_inputs(building_blocks, is_shuffle, bound_test=None, seed=seed, is_reverse=is_reverse, eos=eos)
    _check_sizes(building_blocks, n_inputs, max_composition_train, n_unary_tables, n_heldout_tables, n_heldout_compositions, n_heldout_inputs)
    if bound_test is not None:
        # bound only testing sets
        building_blocks[2:] = [df.iloc[:bound_test] for df in building_blocks[2:]]
    unary_functions, multiary_train, heldout_inputs, heldout_compositions, heldout_tables, new_compositions = building_blocks

    # validation
    multiary_train, validation = _uniform_split(multiary_train, names_unary_train, validation_size=validation_size, seed=seed)
    train = pd.concat([unary_functions, multiary_train], axis=0)

    out = (train, validation, heldout_inputs, heldout_compositions, heldout_tables, new_compositions,
           longer_seens, longer_incrementals, longer_news)

    # adds target attention
    if is_target_attention:
        out = [_append_target_attention(o, eos, is_reverse) for o in out[:-3]]
        for longer in (longer_seens, longer_incrementals, longer_news):
            out.append([_append_target_attention(o, eos, is_reverse) for o in longer])

    return out


def create_N_table_lookup(N=None,
                          alphabet=['0', '1'],
                          n_repeats=3,
                          namer=lambda i: "t{}".format(i + 1),
                          seed=123):
    """Create N possible table lookups.

    Args:
        N (int, optional): number of tables lookups to create. (default: all posible)
        alphabet (list of char, optional): possible characters given as input.
        n_repeats (int, optional): number of characters in `alphabet` used in each input and output.
        namer (callable, optional): function that names a table given an index.
        seed (int, optional): sets the seed for generating random numbers.

    Returns:
        out (list of pd.Series): list of N dataframe with keys->input, data->output, name->namer(i).
    """
    np.random.seed(seed)
    inputs = np.array(list(''.join(letters)
                           for letters in itertools.product(alphabet, repeat=n_repeats)))
    iter_outputs = itertools.permutations(inputs)
    if N is not None:
        iter_outputs = np.array(list(iter_outputs))
        indices = np.random.choice(range(len(iter_outputs)), size=N, replace=False)
        iter_outputs = iter_outputs[indices]
    return [pd.Series(data=outputs, index=inputs, name=namer(i)) for i, outputs in enumerate(iter_outputs)]


def compose_table_lookups(table1, table2, is_intermediate=True):
    """Create a new table lookup as table1 ∘ table2."""
    left = table1.to_frame()
    right = table2.to_frame()
    right['next_input'] = right.iloc[:, 0].str.split().str[-1]
    merged_df = pd.merge(left, right, left_index=True, right_on='next_input').drop("next_input", axis=1)
    left_col, right_col = merged_df.columns

    if is_intermediate:
        merged_serie = merged_df[right_col] + " " + merged_df[left_col]
    else:
        merged_serie = merged_df[left_col]

    merged_serie.name = " ".join([left_col.split("_")[0], right_col.split("_")[0]])

    return merged_serie


def format_input(table, is_copy_input=True, is_reverse=False, eos=None):
    """Formats the input of the task.

    Args:
        table (pd.Series, optional): Serie where keys->input, data->output, name->namer(i)
        is_copy_input (bool, optional): whether to have the inputs first and then the tables. Ex: if reverse:
            001 t1 t2 else t2 t1 001.
        is_reverse (bool, optional): whether to reverse the  input sequence to match the mathematical composition.
            I.e if given, then uses `t1(t2(input))` without parenthesis instead of `input t2 t1`.'
        eos (str, optional): str to append at the end of each input.

    Returns:
        out (pd.Series): Serie where keys->input+name, data->output, name->namer(i).
    """
    inputs = table.index

    table.index = ["{} {}".format(table.name, i) for i in table.index]

    if not is_reverse:
        table.index = [" ".join(i.split()[::-1]) for i in table.index]

    if eos is not None:
        table.index = ["{} {}".format(i, eos) for i in table.index]

    if is_copy_input:
        table.iloc[:] = inputs + " " + table

    return table


### HELPERS ###
def _save_tsv(data, name, path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    if isinstance(data, list):
        for i, df in enumerate(data):
            df.to_csv(os.path.join(path, "{}_{}.tsv".format(name, i + 1)),
                      header=False,
                      sep=str('\t'))  # wraps sep around str for python 2
    else:
        data.to_csv(os.path.join(path, "{}.tsv".format(name)),
                    header=False,
                    sep=str('\t'))  # wraps sep around str for python 2


def flatten(l):
    if isinstance(l, list):
        return reduce(operator.add, l)
    else:
        return l


def assert_equal(a, b):
    assert a == b, "{} != {}".format(a, b)


def _split_seen_unseen_new(dfs, name_train, name_test):
    """Split list of datatframes such that `seen` has only tables in `name_train`, `new` has only tables
    in `name_test`, and the rest is in `unseen`."""

    def _table_is_composed_of(composed_table, tables):
        return set(composed_table.name.split()).intersection(tables)

    seen = [t for t in dfs if not _table_is_composed_of(t, name_test)]
    new = [t for t in dfs if not _table_is_composed_of(t, name_train)]
    unseen = [t for t in dfs if _table_is_composed_of(t, name_test) and _table_is_composed_of(t, name_train)]
    return seen, unseen, new


def _merge_format_inputs(list_dfs, is_shuffle, bound_test=None, seed=None, **kwargs):
    if list_dfs == []:
        return []

    list_df = [pd.concat([format_input(df, **kwargs) for df in dfs],
                         axis=0)
               if dfs != [] else pd.DataFrame()
               for dfs in list_dfs]

    if is_shuffle:
        list_df = [df.sample(frac=1, random_state=seed) for df in list_df]

    if bound_test is not None:
        # better to use is_shuffle when bounding test
        list_df = [df.iloc[:bound_test] for df in list_df]

    return list_df


def _uniform_split(to_split, table_names, validation_size=0.1, seed=None, is_stratify=True):
    df = to_split.to_frame()
    for name in table_names:
        df[name] = [name in i.split() for i in df.index]
    df['length'] = [len(i.split()) for i in df.index]

    stratify = df.iloc[:, 1:] if is_stratify else None

    try:
        train, test = train_test_split(to_split, test_size=validation_size, random_state=seed, stratify=stratify)
    except ValueError:
        warnings.warn("Doesn't use stratfy as given validation_size was to small.")
        train, test = train_test_split(to_split, test_size=validation_size, random_state=seed, stratify=None)

    return train, test


def _check_sizes(dfs, n_inputs, max_length, n_unary_tables, n_heldout_tables, n_heldout_compositions, n_heldout_inputs):
    unary_functions, multiary_train, heldout_inputs, heldout_compositions, heldout_tables, new_compositions = dfs

    # n_inputs is alphabet**n_repeats
    n_train_tables = n_unary_tables - n_heldout_tables
    n_train_compositions = sum(n_train_tables**i for i in range(2, max_length + 1)) - n_heldout_compositions

    def _size_compose(n_tables):
        return n_tables**max_length * n_inputs

    assert_equal(len(unary_functions), n_unary_tables * n_inputs)
    assert_equal(len(heldout_inputs), (n_train_tables**max_length - n_heldout_compositions) * n_heldout_inputs)
    assert_equal(len(multiary_train), n_train_compositions * n_inputs - len(heldout_inputs))
    assert_equal(len(heldout_compositions), n_heldout_compositions * n_inputs)
    assert_equal(len(heldout_tables),
                 _size_compose(n_train_tables + n_heldout_tables) - _size_compose(n_train_tables) - _size_compose(n_heldout_tables))
    assert_equal(len(new_compositions), _size_compose(n_heldout_tables))


def _append_target_attention(df, eos, is_reverse):
    """Appends the target attention by returning a datfarme with the attention given a series."""
    def _len_no_eos(s):
        return len([el for el in s.split() if el != eos])

    df = df.to_frame()
    df["taget attention"] = [" ".join(str(i) for i in range(_len_no_eos(inp))) for inp in df.index]
    if is_reverse:
        df["taget attention"] = [ta[::-1] for ta in df["taget attention"]]
    if eos != "":
        df["taget attention"] = [ta + " " + str(len(ta.split())) for ta in df["taget attention"]]
    return df


### SCRIPT ###
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
