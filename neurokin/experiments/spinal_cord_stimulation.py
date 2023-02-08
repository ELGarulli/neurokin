from neurokin.utils.neural import processing
from numpy.typing import ArrayLike


class SpinalCordStimulation:

    def __init__(self, neural_data):
        self.stimulation_timestamps: ArrayLike
        self.sync_data = neural_data.sync_data

    def set_stimulation_timestamps(self, expected_pulses):
        self.stimulation_timestamps = processing.get_stim_timestamps(self.sync_data,
                                                                     expected_pulses=expected_pulses)

    def sensory_evoked_potential_analysis(self, channel, start_window, end_window, pulse_number,
                                          amplitude_succession_protocol):
        channel_raw = self.raw[channel]
        parsed_sep = processing.parse_raw(channel_raw, self.stim_timestamps, start_window, end_window)
        avg_amplitudes = processing.get_average_amplitudes(parsed_sep, amplitude_succession_protocol, pulse_number)
        return avg_amplitudes


