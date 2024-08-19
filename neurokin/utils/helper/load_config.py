import yaml


# TESTME with config file
def read_config(path, converts_keys_to_string=False):
    """
    Reads structured config file defining a project.

    :param path: path to config
    :param converts_keys_to_string:
    :return: dict from a yaml file
    """

    try:
        with open(path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find the config file at " + path + " \n Please make sure the path is correct and the file exists")

    if converts_keys_to_string:
        keys2string(cfg)

    return cfg


def keys2string(d):
    """
    Convert all keys in strings (helpful if dates are keys).

    Reference: https://stackoverflow.com/questions/62198378/numeric-keys-in-yaml-files
    :param d: dictionary to convert the keys of
    :return: converted dictionary
    """
    if isinstance(d, dict):
        for idx, k in enumerate(list(d.keys())):
            if not isinstance(k, str):
                sk = str(k)
                d[sk] = d.pop(k)
                k = sk
            keys2string(d[k])
    elif isinstance(d, list):
        for e in d:
            keys2string(e)
