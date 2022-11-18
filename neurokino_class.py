from neural_data import NeuralData
from kinematic_data import KinematicDataRun
from utils.neurokino_helper import (convert_start_end_between_two_fs, compute_spectrograms_steps,
                                    average_spectrograms, get_idx_for_pad_steps)


class Neurokino:
    """
    This class represents a full experiment with neural and kinematic data. E.g. a single run.
    """

    def __init__(self, neural_object: NeuralData, kinematic_object: KinematicDataRun):
        self.neural_object = neural_object
        self.kinematic_object = kinematic_object
        self.standardized_steps
    def get_steps_neural_correlates(self, channel, side, nperseg=None, noverlap=None):
        """
        Given a neural and a kinematic object containing information on the steps. It returns the average spectrograms
        centered on the maximum of the steps and the average step.
        :param channel: the channel of the raw signal for which to compute the spectrogram for
        :return: the average spectrograms and the average step
        #TODO add support for single kinematic side
        #TODO add support for two neural sides
        #TODO should it support/return parsed kinematics steps or out of scope
        """

        if side.lower() == "left" or side.lower() == "l":
            starts = self.kinematic_object.left_mtp_lift
            maxs = self.kinematic_object.left_mtp_max
            ends = self.kinematic_object.left_mtp_land
        elif side.lower() == "right" or side.lower() == "r":
            starts = self.kinematic_object.right_mtp_lift
            maxs = self.kinematic_object.right_mtp_max
            ends = self.kinematic_object.right_mtp_land
        else:
            raise ValueError("Please select a valid side, either left, right, l or r. "
                             "Case insensitive.")

        padded_steps = get_idx_for_pad_steps(steps_start_idxs=starts,
                                             steps_max_idxs=maxs,
                                             steps_stop_idxs=ends)

        neural_step_bounds = convert_start_end_between_two_fs(padded_steps,
                                                              first_frame=self.kinematic_object.gait_cycles_start,
                                                              origin_fs=self.kinematic_object.fs,
                                                              result_fs=self.neural_object.fs)

        spectrograms = compute_spectrograms_steps(raw=self.neural_object.raw[channel],
                                                  fs=self.neural_object.fs,
                                                  steps_idxs=neural_step_bounds,
                                                  nfft=nperseg,
                                                  noverlap=noverlap)

        avg_spectrogram = average_spectrograms(spectrograms)

        return avg_spectrogram
