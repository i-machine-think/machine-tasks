from utils.get_default_params import get_default_params
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_symbol_rewriting_data(is_small=False, is_mini=False):
    """
    Gets the symbol rewriting data in task argument format
    Args:
        is_small (bool, optional) - Whether to make default params have lower default k 
        is_mini (bool, optional) - Whether to make default params be optimized for training speed
                                 - k=1;batch_size=128;patience=2;epochs=3
    Returns: 
        Arguments for Task object
            - "Symbol Rewriting" (task name)
            - data_dir, (directory of symbol rewriting/data files)
            - train_file (str) - name of training file
            - valid_file (str) - name of validation file
            - test_files (str) - name of test file
            - default_params (must be present as json in SymbolRewriting folder) 
                - offers training suggestions
    """

    train_file = "grammar_std.train.full"
    test_files = ["grammar_long.tst.full", "grammar_repeat.tst.full",
                  "grammar_short.tst.full", "grammar_std.tst.full"]
    valid_file = "grammar.val"
    data_dir = os.path.join(dir_path, "data")

    # Get default params from json
    # - these are not required but offer recommendation on default params
    default_params = get_default_params(dir_path)

    if is_small:
        train_file = "grammar_std.train.small"
        default_params["task_defaults"]["k"] = 1
    if is_mini:
        train_file = "grammar_std.train.small"
        default_params["task_defaults"]["k"] = 1
        default_params["task_defaults"]["batch_size"] = 128
        default_params["task_defaults"]["patience"] = 2
        default_params["task_defaults"]["epochs"] = 3
        default_params["task_defaults"]["n_attn_plots"] = 1

    return ("Symbol Rewriting", data_dir, train_file,
            valid_file, test_files, default_params)
