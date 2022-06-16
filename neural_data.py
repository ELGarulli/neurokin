import tdt
import json
import numpy as np
from typing import Dict


def get_data_array(array, channel_number):
    s_number = int(len(array) / channel_number)
    a = np.zeros(shape=(channel_number, s_number))
    for i in range(channel_number):
        for j, z in zip(range(0, len(array), channel_number), range(s_number)):
            a[i][z] = array[i + j]
    return a


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

        self.raw: Dict[int, np.ndarray]
        self.fs: float
        self.raw, self.fs = self.load_neural_data()

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
        # note the path should have the record note
        recording_path = self.path
        structure_path = recording_path + "/structure.oebin"
        with open(structure_path) as f:
            structure = json.load(f)

            #TODO refactor to have path in consts
        source_processor_id = structure["source_processor_name"][0]["source_processor_name"].replace(" ", "-") + \
                              structure["source_processor_name"][0]["recorded_processor_id"] + ".0"

        binary_data_path = recording_path + "experiment1/recording1/continuous" + source_processor_id + '/continuous.dat'
        fs = structure["continuous"][0]["sample_rate"]

        n_ch = structure["continuous"][0]["num_channels"]
        neural_data_bin = np.fromfile(binary_data_path + '/continuous.dat', dtype='<i2')
        neural_data_au = get_data_array(array=neural_data_bin, channel_number=n_ch)

        return neural_data_au, fs

    def compute_psd(self):
        psd = None
        freqs = None
        t = None
        return
