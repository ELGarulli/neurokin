from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from utils.neural.processing import get_spectrogram_data


def convert_frame_between_two_fs(frame: int, first_frame: int, origin_fs: float,
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


def convert_start_end_between_two_fs(frames: ArrayLike, first_frame: int,
                                     origin_fs: float, result_fs: float):
    """
    Wrapper around the single frame conversion for convenient conversion of teh step bounds.
    See convert_frame_between_two_fs for more info.
    :param frames: arraylike of shape N*(start, stop)
    :param first_frame: initial frame in the original fs
    :param origin_fs: sampling frequency in the original system
    :param result_fs: sampling frequency in the resulting system
    :return: arraylike of shape N*(start, stop) with converted frames
    """
    converted = []
    for bounds in frames:
        start = convert_frame_between_two_fs(bounds[0], first_frame, origin_fs, result_fs)
        end = convert_frame_between_two_fs(bounds[1], first_frame, origin_fs, result_fs)

        converted.append((start, end))
    return converted


def compute_spectrograms_steps(raw, fs, steps_idxs, **kwargs):
    """
    For each step it computes the spectrograma and returns the values, the list of frequencies and times
    :param raw: neural array
    :param fs: sampling frequency
    :param steps_idxs: bounds of each step
    :param kwargs: eg nperseg and noverlap
    :return: list of tuples (pxx, freq, t)
    """
    spectrograms = []
    for i in range(len(steps_idxs)):
        start = steps_idxs[i][0]
        end = steps_idxs[i][-1]
        neural_step = raw[start:end]
        pxx, freq, t = get_spectrogram_data(fs=fs, raw=neural_step, **kwargs)
        spectrograms.append((pxx, freq, t))
    return spectrograms


def average_spectrograms(spectrograms):
    """
    Computes the averaged spectrogram
    :return:
    """
    freq = np.asarray(spectrograms[0][1])
    t = np.asarray(spectrograms[0][2])
    pxx = []
    for s in range(len(spectrograms)):
        if not (freq == np.asarray(spectrograms[s][1])).all():
            print("WARNING: frequencies don't have the same value")
        if not (t == np.asarray(spectrograms[s][2])).all():
            print("WARNING: time doesn't have the same values")
        pxx.append(spectrograms[s][0])

    pxx = np.asarray(pxx)
    pxx_avg = np.mean(pxx, axis=0)
    return pxx_avg, freq, t


def get_idx_for_pad_steps(steps_start_idxs, steps_max_idxs, steps_stop_idxs):
    """
    Shifts the bounds of the steps to be the length of the longest one
    :param steps_start_idxs: list of start of each step
    :param steps_max_idxs: list of max elevations of each step
    :param steps_stop_idxs: list of end of each step
    :return:
    """
    max_len_to_peak = np.max(np.asarray([max_ - start for start, max_ in zip(steps_start_idxs, steps_max_idxs)]))
    max_len_step = np.max(np.asarray([end - start for end, start in zip(steps_stop_idxs, steps_start_idxs)]))

    padded_steps = []

    for step in range(len(steps_stop_idxs)):
        diff_to_peak = max_len_to_peak - abs(steps_max_idxs[step] - steps_start_idxs[step])
        start = steps_start_idxs[step] - diff_to_peak
        # TODO think of a way to make robust. what if 0? still have neural data but should allow?
        end = start + max_len_step
        padded_steps.append((start, end))
    return padded_steps
