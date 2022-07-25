import numpy as np
from utils import importing
from utils import exporting



class NeuralData:
    """
    This class represents the neural data recording of a single recording experiment.
    It allows to load neural data from multiple recording systems and to perform basic analysis on the raw data.
    """

    def __init__(self, path, config_file):
        self.path = path
        self.config_file = config_file
        self.config = self.load_config_file()

        self.run_id = self.config["run_id"]

        self.raw: np.ndarray
        self.fs: float
        self.raw, self.fs = self.load_neural_data()

    def load_config_file(self):
        # config = load config_file
        # TODO load config_file in dict
        return {}

    def load_tdt_data(self):
        self.fs, self.raw = importing.import_tdt_channel_data(folderpath=self.path)

    def load_open_ephys(self):
        #TODO refactor name data
        self.fs, self.raw = importing.import_open_ephys_channel_data(folderpath=self.path)

    def export_neural_data_to_binary(self, filename):
        exporting(self.raw, filename)