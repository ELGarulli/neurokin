import tdt
import numpy as np
from typing import Dict


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

        self.data: Dict[int, np.ndarray]
        self.fs: float
        self.data, self.fs = self.load_neural_data()

    def load_neural_data(self):
        # TODO load differently for tdt and open_ephys
        recording_system = self.config["recording_system"]
        if recording_system == "tdt":
            data = self.load_tdt()
        elif recording_system == "open_ephys":
            data = self.load_open_ephys()
        else:
            raise
        # TODO find good way to raise "recording system not recognised error"
        return data

    def load_config_file(self):
        # config = load config_file
        # TODO load config_file in dict
        return {}

    def load_tdt(self):
        # TODO load neural data in dict
        data = tdt.read_block(self.path)  # import data
        try:
            raw = data.streams.LFP1.data  # get the raw data out of the inner pipeline structure
            fs = data.streams.LFP1.fs  # sampling frequency
        except AttributeError:
            try:
                raw = data.streams.EOG1.data  # get the raw data out of the inner pipeline structure
                fs = data.streams.EOG1.fs  # sampling frequency
            except AttributeError:
                try:
                    raw = data.streams.NPr1.data  # get the raw data out of the inner pipeline structure
                    fs = data.streams.NPr1.fs  # sampling frequency
                except AttributeError:
                    raise
        return raw, fs

    def load_open_ephys(self):
        # TODO load neural data in dict
        return {}

    def compute_psd(self):
        psd = None
        freqs = None
        t = None
        return
