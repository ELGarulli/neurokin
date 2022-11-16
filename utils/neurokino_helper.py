from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike



def get_kinematic_steps():
    """
    Retrieves the kinematics data section corresponding to the kinematic steps
    :return: 2D arrays of parsed neural data L and R
    """
    return


def convert_frames_between_two_fs(frame: int, first_frame: int, origin_fs: float,
                                  result_fs: float):
    """
    Converts the index between two different sampling frequency.
    Requires to explicitly set the first frame to avoid undetectable data shifts.
    :param frame: index to convert
    :param first_frame: initial frame in the original fs
    :param origin_fs: sampling frequency in the original system
    :param result_fs: sampling frequency in the resulting system
    :return: converted index
    """
    first_frame_neural = first_frame * result_fs / origin_fs
    frame_neural = int(first_frame_neural + frame * result_fs / origin_fs)

    return frame_neural


#def parse_neural_steps(neural_raw: ArrayLike, idxs: Tuple[int, int]):
#    """
#    Given the steps indexes it returns the neural data corresponding to each step
#    as N*M 2D where N=number of steps and M = max step len
#    :param neural_raw: raw neural recording from a channel
#    :param idxs: ´tuple containing start and end index
#    :return:
#    """
#    parsed_steps = nu
#    return


def center_spectrograms():
    """
    Centers the steps around the maximum elevation in the steps (assuming to work with MTP traces or equivalent).

    #TODO check how julie did
    :return:
    """
    return


def compute_spectrogram():
    # TODO necessary? or redundant with analogus neural function?
    return


def average_spectrograms():
    """
    Computes the averaged spectrogram
    :return:
    """
    return
