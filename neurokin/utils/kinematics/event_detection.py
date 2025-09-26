from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy import signal


def get_toe_lift_landing(y: ArrayLike, recording_fs: float, step_filter_freq: int, prominence: float,
                         relative_height: float):
    """
    Returns the left and right bounds of the gait cycle, corresponding to the toe lift off and the heel strike.
    As a first step it smooths the signal to increase robust peak detection.

    :param y: trace of the toe in the z coordinate
    :param recording_fs: sampling rate of the kinematics recording
    :param step_filter_freq: used to filter out very jittery movement which should not represent steps
    :param prominence: required minimal prominence of peaks
    :param relative_height: Chooses the relative height at which the peak width is measured as a percentage of
        its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half
        the prominence height. Must be at least 0.
    :return: left, right bounds and max value of each peak.
    """
    y = lowpass_array(y, step_filter_freq, recording_fs)

    max_x, _ = signal.find_peaks(y, prominence=prominence)
    avg_distance = abs(int(median_distance(max_x) / 2))

    lb = []
    rb = []

    for p in max_x:
        left = p - avg_distance if p - avg_distance > 0 else 0
        right = p + avg_distance if p + avg_distance < len(y) else len(y)
        bounds = get_peak_boundaries_scipy(y=y[left:right], px=p, left_crop=left, relative_height=relative_height)
        lb.append(bounds[0])
        rb.append(bounds[1])

    lb = np.asarray(lb)
    rb = np.asarray(rb)
    return lb, rb, max_x


def get_peak_boundaries_scipy(y: ndarray, px: float, left_crop: int, relative_height: float) -> Tuple[int, int]:
    """
    Computes the boundaries of a step by getting the peaks boundaries

    :param y: y values
    :param px: peaks indexes
    :param left_crop: how many samples to use to crop on the left side
    :param relative_height: Chooses the relative height at which the peak width is measured as a percentage of
        its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half
        the prominence height. Must be at least 0.
    :return: returns boundaries of steps
    """
    peaks = np.asarray([px - left_crop])
    peak_pro = signal.peak_prominences(y, peaks)
    peaks_width = signal.peak_widths(y, peaks, rel_height=relative_height, prominence_data=peak_pro)
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
