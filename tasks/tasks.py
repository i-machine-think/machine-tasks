
from LongLookupTables.get_long_lookup_tables import get_long_lookup_tables
from LookupTables.get_lookup_tables import get_lookup_tables
from SymbolRewriting.get_symbol_rewriting_data import get_symbol_rewriting_data

import os


def get_task(name,
             is_small=False,
             is_mini=False,
             longer_repeat=5):
    """Return the wanted tasks.

    Args:
        name ({"lookup", "long_lookup", "long_lookup_oneshot",
            "long_lookup_reverse", "noisy_long_lookup_multi", "noisy_long_lookup_single",
            "long_lookup_intermediate_noise", "symbol_rewriting", "SCAN"}) 
            name of the task to get.
        is_small (bool, optional): whether to run a smaller verson of the task.
            Used for getting less statistically significant results.
        is_mini (bool, optional): whether to run a smaller verson of the task.
            Used for testing purposes.
        longer_repeat (int, optional): number of longer test sets.

    Returns:
        task (tasks.tasks.Task): instantiated task.
    """
    name = name.lower()

    # classical lookup table
    if name == "lookup":
        task_args = get_lookup_tables(is_small=is_small, is_mini=is_mini)

    # Long lookup tasks - paser in get_long_lookup_tables can figure out which
    elif "lookup" in name:
        task_args = get_long_lookup_tables(
            name, is_small=is_small, is_mini=is_mini,
            longer_repeat=longer_repeat)

    # classical symbol rewriting task
    elif name == "symbol_rewriting":
        task_args = get_symbol_rewriting_data(
            is_small=is_small, is_mini=is_mini)

        # classical scan
    elif name == "SCAN":
        raise NotImplementedError(
            "SCAN dataset not yet implemented to be used as Task Object")

    else:
        raise ValueError("Unkown name : {}".format(name))

    return Task(*task_args)


class Task(object):
    """Helper class containing meta information of datasets.

    Args:
        name (str): name of the dataset.
        data_dir (str):  directory to prepend to all path above.
        train_path (str): path to training data.
        test_paths (list of str): list of paths to all test data.
        valid_path (str): path to validation data.
        default_params (None or Dict): default params that represent baseline
        extension (str, optional): extension to add to every paths above.
    """

    def __init__(self,
                 name,
                 data_dir,
                 train_filename,
                 valid_filename,
                 test_filenames,
                 default_params,
                 extension="tsv"):

        self.name = name
        self.extension = "." + extension
        self.data_dir = data_dir
        self.train_path = self._add_presufixes(train_filename)
        self.valid_path = self._add_presufixes(valid_filename)
        self.test_paths = [self._add_presufixes(
            path) for path in test_filenames]

        self.default_params = default_params

        self._validate_all_filepaths()

    def _add_presufixes(self, path):
        if path is None:
            return None
        return os.path.join(self.data_dir, path) + self.extension

    def __repr__(self):
        return "{} Task".format(self.name)

    def _validate_all_filepaths(self):
        """
        Returns Error if a path is invalid in the stored paths
        """
        paths = [self.train_path, self.valid_path] + self.test_paths

        for p in paths:
            if not os.path.isfile(p):
                raise NameError("File at {} does not exist".format(p))
