from typing import Tuple
from numpy import ndarray
from scipy import signal
from constants.gait_cycle_detection import RELATIVE_HEIGHT
from matplotlib import pyplot as plt

import numpy as np


def get_toe_lift_landing(y):
    """
    Returns the left and right bounds of the gait cycle, corresponding to the toe lift off and the heel strike.
    :param y: trace of the toe in the z coordinate
    :return: left and right bounds
    """
    max_x, _ = signal.find_peaks(y, prominence=1)
    avg_distance = abs(int(median_distance(max_x) / 2))
    lb = []
    rb = []

    for p in max_x:
        left = p - avg_distance if p - avg_distance > 0 else 0
        right = p + avg_distance if p + avg_distance < len(y) else len(y)
        bounds = get_peak_boundaries_scipy(y=y[left:right], px=p, left_crop=left)
        lb.append(bounds[0])
        rb.append(bounds[1])

    return lb, rb, max_x


def get_peak_boundaries_scipy(y: ndarray, px: float, left_crop: int) -> Tuple[int, int]:
    peaks = np.asarray([px - left_crop])
    peak_pro = signal.peak_prominences(y, peaks)
    peaks_width = signal.peak_widths(y, peaks, rel_height=RELATIVE_HEIGHT, prominence_data=peak_pro)
    intersections = peaks_width[-2:]
    try:
        left = intersections[0] + left_crop
    except:
        left = left_crop

    try:
        right = intersections[-1] + left_crop
    except:
        right = len(y) + left_crop

    return [left, right]


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