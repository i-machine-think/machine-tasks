# import tqdm
import os
import json

from LongLookupTables.make_long_lookup_tables import make_long_lookup_tables
from utils.get_default_params import get_default_params

dir_path = os.path.dirname(os.path.realpath(__file__))

# Dictionary that maps input name of lookup table dataset
# to directory where it is located
name2dir = {
    # Lookup tables with training up to 3 compositions
    "long_lookup": "LongLookupTables",

    # Lookup tables with training 1 2 and 4 compositions (i.e jumping 3)
    # - currently does not have appropriate generation.txt
    # "long lookup jump": "LongLookupTablesJump",

    # long lookup tables with a iniital training file without t7 and t8
    # and then adding uncomposed t7 and t8 with all the rest
    "long_lookup_oneshot": "LongLookupTablesOneShot",

    # reverse long lookup table (i.e right to left hard attention)
    "long_lookup_reverse": "LongLookupTablesReverse",

    # noisy long lookup table where between each "real" table there's one noisy
    # one. THe hard attention is thus a diagonal wich is less steep
    "long_lookup_intermediate_noise": "LongLookupTablesIntermediateNoise",

    # noisy long lookup table with a special start token saying when are the
    # "real tables" starting. THe hard attention is thus a diagonal that
    # starts at some random position.
    "noisy_long_lookup_single": "NoisyLongLookupTablesSingle",

    # noisy long lookup table where there are multiple start token and only
    # the last one really counts
    "noisy_long_lookup_multi": "NoisyLongLookupTablesMulti"
}


def get_long_lookup_tables(name, is_small=False, is_mini=False, longer_repeat=5):
    """
    Return the wanted lookup dataset information, downloads or generates
    it if it is not already present

    Args:
        name ({"long_lookup", "long_lookup_oneshot", "long_lookup_reverse", 
            "noisy_long_lookup_multi", "noisy_long_lookup_single",
            "long_lookup_intermediate_noise"}) name of the long lookup task to get.

        is_small (bool, optional): whether to run a smaller verson of the task.
            Used for getting less statistically significant results.
        is_mini (bool, optional): whether to run a smaller verson of the task.
            Used for testing purposes.
        longer_repeat (int, optional): number of longer test sets. 
            - note if data is already generated with a certain longer repeat 
            - If a longer repeat is called then the return paths will not exist
            - In this case either delete the data folder from the specific LongLookup set
            - Or implement a check to extend it and regenerate with higher longer_repeat

    Returns:
        task arguments (to be passed to instantiate a Task Object)
    """

    print("Getting Lookup Table")
    name = name.lower()
    lookup_tables_dir_path = os.path.join(dir_path, name2dir[name])
    if not os.path.isdir(lookup_tables_dir_path):
        raise NotImplementedError(
            "Folder at {} does not exist".format(lookup_tables_dir_path))

    generation_arguments_path = os.path.join(
        lookup_tables_dir_path, 'generation_arguments.txt')
    if not os.path.isfile(generation_arguments_path):
        raise NotImplementedError(
            "Generation Arguments .txt Missing in Table Lookup Folder \
             - Cannot Generate Table")

    lookup_tables_data_dir_path = os.path.join(lookup_tables_dir_path, "data")

    if not os.path.isdir(lookup_tables_data_dir_path):
        print("Data not present for {}".format(name))
        print("Generating Dataset")
        make_long_lookup_tables(
            lookup_tables_data_dir_path, generation_arguments_path)

    # Get default params from json
    # - these are not required but offer recommendation on default params
    default_params = get_default_params(lookup_tables_dir_path)

    # Update the defauls params if task is small /mini
    if default_params is not None:
        if is_small:
            default_params["task_defaults"]["k"] = 1
        if is_mini:
            default_params["task_defaults"]["k"] = 1
            default_params["task_defaults"]["batch_size"] = 128
            default_params["task_defaults"]["patience"] = 2
            default_params["task_defaults"]["epochs"] = 3
            default_params["task_defaults"]["n_attn_plots"] = 1

    train_file = "train"
    valid_file = "validation"
    test_files = flatten(["heldout_inputs", "heldout_compositions",
                          "heldout_tables",
                          "new_compositions", repeat(
                              "longer_seen", longer_repeat),
                          repeat("longer_incremental", longer_repeat),
                          repeat("longer_new", longer_repeat)])

    return (name, lookup_tables_data_dir_path,
            train_file, valid_file, test_files, default_params)


# Helper Functions


def flatten(l):
    """Flattens a list of element or lists into a list of elements."""
    out = []
    for e in l:
        if not isinstance(e, list):
            e = [e]
        out.extend(e)
    return out


def repeat(s, n, start=1):
    """Repeats a string multiple times by adding a iter index to the name."""
    return ["{}_{}".format(s, i) for i in range(start, n + start)]


def filter_dict(d, remove):
    """Filters our the key of a dictionary."""
    return {k: v for k, v in d.items() if k not in remove}
