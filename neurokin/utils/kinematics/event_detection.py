from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy import signal

from neurokin.constants.gait_cycle_detection import RELATIVE_HEIGHT, STEP_FILTER_FREQ, PROMINENCE


# TESTME with pickled steps data
def get_toe_lift_landing(y, recording_fs):
    """
    Returns the left and right bounds of the gait cycle, corresponding to the toe lift off and the heel strike.
    As a first step it smooths the signal to increase robust peak detection.

    :param y: trace of the toe in the z coordinate
    :param recording_fs: sampling rate of the kinematics recording
    :return: left, right bounds and max value of each peak.
    """
    y = lowpass_array(y, STEP_FILTER_FREQ, recording_fs)

    max_x, _ = signal.find_peaks(y, prominence=PROMINENCE)
    avg_distance = abs(int(median_distance(max_x) / 2))

    lb = np.where(max_x - avg_distance > 0, max_x - avg_distance, 0)
    rb = np.where(max_x + avg_distance < len(y), max_x + avg_distance, len(y))

    for i, p in enumerate(max_x):
        bounds = get_peak_boundaries_scipy(y=y[lb[i]:rb[i]], px=p, left_crop=lb[i])

    return lb, rb, max_x


# TESTME with pickled steps data
def get_peak_boundaries_scipy(y: ndarray, px: float, left_crop: int) -> Tuple[int, int]:
    """
    Computes the boundaries of a step by getting the peaks boundaries

    :param y: y values
    :param px: peaks indexes
    :param left_crop: how many samples to use to crop on the left side
    :return: returns boundaries of steps
    """
    peaks = np.asarray([px - left_crop])
    peak_pro = signal.peak_prominences(y, peaks)
    peaks_width = signal.peak_widths(y, peaks, rel_height=RELATIVE_HEIGHT, prominence_data=peak_pro)
    intersections = peaks_width[-2:]
    try:
        left = int(intersections[0] + left_crop)
    except:
        left = int(left_crop)

    try:
        right = int(intersections[-1] + left_crop)
    except:
        right = int(len(y) + left_crop)

    return left, right


# TESTME
def lowpass_array(array, critical_freq, fs):
    """
    Low passes the array for a given frequency using a 2nd order butterworth filter and filtfilt to avoid phase shift.

    :param array: input array
    :param critical_freq: critical filter frequency
    :param fs: sampling frequency
    :return: filtered array
    """
    b, a = signal.butter(2, critical_freq, "low", output="ba", fs=fs)
    filtered = signal.filtfilt(b, a, array)
    return filtered


# TESTME
def median_distance(a: ndarray) -> ndarray:
    """
    Gets median distance between peaks

    :param a:
    :return: median
    """
    if np.issubdtype(a.dtype, np.number):
        return np.median(-np.diff(a))
    else:
        raise TypeError("TypeError: unsupported operand type(s) for -:" + str(a.dtype))
