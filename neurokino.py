from neural_data import NeuralData
from kinematic_data import KinematicDataRun
from utils.neurokino_helper import (convert_frames_between_two_fs, compute_spectrogram, center_spectrograms,
                                    average_spectrograms)


class Neurokino:
    """
    This class represents a full experiment with neural and kinematic data. E.g. a single run.
    """

    def __init__(self, neural_object: NeuralData, kinematic_object: KinematicDataRun):
        self.neural_object = neural_object
        self.kinematic_object = kinematic_object

    def get_steps_neural_correlates(self, channel):
        """
        Given a neural and a kinematic object containing information on the steps. It returns the average spectrograms
        centered on the maximum of the steps and the average step.
        :param channel: the channel of the raw signal for which to compute the spectrogram for
        :return: the average spectrograms and the average step
        #TODO add support for single kinematic side
        #TODO add support for two neural sides
        #TODO should it support/return parsed kinematics steps or out of scope
        """
        

        convert_frames_between_two_fs()
        center_spectrograms()
        compute_spectrogram()
        spectrograms = average_spectrograms()
        #kinematic_steps = get_kinematic_steps()
        return spectrograms #, kinematic_steps
