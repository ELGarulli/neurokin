from utils import importing
from utils import exporting
import numpy as np


class NeuralData:
    """
    This class represents the neural data recording of a single recording experiment.
    It allows to load neural data from multiple recording systems and to perform basic analysis on the raw data.
    """

    def __init__(self, path, config_file=None):
        self.path = path
        self.config_file = config_file
        self.config = self.load_config_file()

        self.raw: np.ndarray
        self.fs: float

    def load_config_file(self):
        # config = load config_file
        # TODO load config_file in dict
        return {}

    def load_tdt_data(self):
        self.fs, self.raw = importing.import_tdt_channel_data(folderpath=self.path)

    def load_open_ephys(self, experiment, recording, channels=None):
        self.fs, self.raw = importing.import_open_ephys_channel_data(folderpath=self.path,
                                                                     experiment=experiment,
                                                                     recording=recording,
                                                                     channels=channels)

    def export_neural_data_to_binary(self, filename):
        exporting.export_neural_data_to_bin(self.raw, filename)
