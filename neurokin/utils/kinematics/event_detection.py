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

    lb = []
    rb = []

    for p in max_x:
        left = p - avg_distance if p - avg_distance > 0 else 0
        right = p + avg_distance if p + avg_distance < len(y) else len(y)
        bounds = get_peak_boundaries_scipy(y=y[left:right], px=p, left_crop=left)
        lb.append(bounds[0])
        rb.append(bounds[1])

    lb = np.asarray(lb)
    rb = np.asarray(rb)
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
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return np.median(distances)
