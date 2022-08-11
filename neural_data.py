from utils.neural import importing, processing
from utils import exporting
import numpy as np
from numpy.typing import ArrayLike


class NeuralData:
    """
    This class represents the neural data recording of a single recording experiment.
    It allows to load neural data from multiple recording systems and to perform basic analysis on the raw data.
    """

    def __init__(self, path, config_file=None):
        self.path = path
        self.config_file = config_file
        self.config = self.load_config_file()

        self.raw: ArrayLike
        self.fs: float
        self.sync_data: ArrayLike
        self.stimulation_timestamps: ArrayLike

    def load_config_file(self):
        # config = load config_file
        # TODO load config_file in dict
        return {}

    def load_tdt_data(self, sync_ch: False):
        self.fs, self.raw = importing.import_tdt_channel_data(folderpath=self.path)
        if sync_ch:
            self.sync_data = importing.import_tdt_stimulation_data(folderpath=self.path, t1=0, t2=-1)

    def load_open_ephys(self, experiment, recording, channels=None):
        self.fs, self.raw = importing.import_open_ephys_channel_data(folderpath=self.path,
                                                                     experiment=experiment,
                                                                     recording=recording,
                                                                     channels=channels)

    def set_stimulation_timestamps(self, expected_pulses):
        self.stimulation_timestamps = processing.get_stim_timestamps(self.sync_data, expected_pulses=expected_pulses)

    def sensory_evoked_potential_analysis(self, channel, start_window, end_window, pulse_number,
                                          amplitude_succession_protocol):
        channel_raw = self.raw[channel]
        parsed_sep = processing.parse_raw(channel_raw, self.stim_timestamps, start_window, end_window)
        avg_amplitudes = processing.get_average_amplitudes(parsed_sep, amplitude_succession_protocol, pulse_number)
        return avg_amplitudes
