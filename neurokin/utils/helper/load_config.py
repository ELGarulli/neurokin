import yaml

#TESTME with config file
def read_config(path):
    """
    Reads structured config file defining a project.
    """

    try:
        with open(path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        raise FileNotFoundError(
                "Could not find the config file at " + path + " \n Please make sure the path is correct and the file exists")

    return cfg
