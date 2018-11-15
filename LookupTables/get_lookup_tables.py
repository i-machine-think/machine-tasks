from utils.get_default_params import get_default_params
import os
import json

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_lookup_tables(is_small=False, is_mini=False):
    """
    Gets the small lookup table data in task argument format
    Args:
        is_small (bool, optional) - Whether to make default params have lower default k 
        is_mini (bool, optional) - Whether to make default params be optimized for training speed
                                 - k=1;batch_size=128;patience=2;epochs=3
    Returns: 
        Arguments for Task object
            - "lookup" (task name)
            - lookup_tables_data_dir_path, (directory of lookup table data files)
            - train_file (str) - name of training file
            - valid_file (str) - name of validation file
            - test_files (str) - name of test file
            - default_params (None or dictionary if present) - offers training suggestions
    """

    lookup_tables_data_dir_path = os.path.join(dir_path, "data")
    if not os.path.isdir(lookup_tables_data_dir_path):
        print("Data not present for lookup \n Downloading Dataset")
        download_lookup_tables(lookup_tables_data_dir_path)

    data_dir = os.path.join(lookup_tables_data_dir_path, "samples/sample1/")
    test_files = ["heldout_inputs", "heldout_compositions", "heldout_tables",
                  "new_compositions", "longer_compositions_seen",
                  "longer_compositions_incremental", "longer_compositions_new"]
    valid_file = "validation"
    train_file = "train"

    # Get default params from json
    # - these are not required but offer recommendation on default params
    default_params = get_default_params(dir_path)

    # Update the defauls params if task is small /mini
    if is_small:
        default_params["task_defaults"]["k"] = 1
    if is_mini:
        default_params["task_defaults"]["k"] = 1
        default_params["task_defaults"]["batch_size"] = 128
        default_params["task_defaults"]["patience"] = 2
        default_params["task_defaults"]["epochs"] = 3
        default_params["task_defaults"]["n_attn_plots"] = 1

    return ("lookup", data_dir,
            train_file, valid_file, test_files, default_params)


def download_lookup_tables(data_dir_path):
    """
    Downloads the lookup-3bit zip from a url and unzips/untar it into the passed data folder
    Args: 
        data_dir_path (path): where to download and unzip the lookup-3bit data 
    """
    print("Downloading LookupTables-3bit data to {}".format(data_dir_path))
    raise NotImplementedError(
        "Downloading the LookupTable Data is not Implemented")


def validate_all_filepaths(paths):
    """ Returns false if a path is invalid in the list of paths """
    for p in paths:
        if not os.path.isfile(p):
            raise NameError("File at {} does not exist".format(p))
    return True
