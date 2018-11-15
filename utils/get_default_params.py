import json
import os
import warnings


def get_default_params(path):
    """
    Reads the default_params json file in the directory passed in path
    Outputs either - None if .json file does not exist
                   - Or a dictionary object of the json if it does
    """

    path = os.path.join(path, "default_params.json")

    default_params = None
    if not os.path.isfile(path):
        warnings.warn("Default Params File Missing at {} \n \
        but still invoked default_params function".format(path), Warning)
    else:
        with open(path) as json_data:
            default_params = json.load(json_data)

    return default_params
