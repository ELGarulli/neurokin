from typing import Tuple, List
from numpy import ndarray
from scipy import interpolate
from scipy.stats import linregress
from constants.gait_cycle_detection import (LOCAL_POINTS, SLOPE_INITIAL_TILT, TILTING_STEPS, INTERSECTION_NUMBER)
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import statistics
from scipy.signal import find_peaks


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


def get_toe_lift_landing(y):
    """
    Returns the left and right bounds of the gait cycle, corresponding to the toe lift off and the heel strike.
    :param y: trace of the toe in the z coordinate
    :return: left and right bounds
    """
    max_x, _ = find_peaks(y, prominence=1)
    y_g = np.gradient(y, 1)
    avg_distance = abs(int(median_distance(max_x) / 2))
    lb = []
    rb = []

    for p in max_x:
        left = p - avg_distance if p - avg_distance > 0 else 0
        right = p + avg_distance if p + avg_distance < len(y_g) else len(y_g)
        bounds = get_peak_boundaries(y=y_g[left:right], px=p, left_crop=left)
        lb.append(bounds[0])
        rb.append(bounds[1])

    return lb, rb, max_x


def get_peak_boundaries(y: ndarray, px: float, left_crop: int) -> Tuple[int, int]:
    """
    Computes the left and right bounds of a peak, analysing the first derivative.
    First it interpolates the signal to be able to use it as a function.
    Then fixes a line through the point corresponding to the max of the peak in the
    original signal. The line is initialized close to a slope computed locally around the point.
    Finally it's rotated contra clock wise, as long as it has three intersection points with
    the interpolated functions.
    :param y: first derivative of the signal
    :param px: x of the maxima point
    :param left_crop: left cropping of the signal (for shifting)
    :return: left and right intersection
    """
    x = np.arange(0, len(y), 1, dtype=float)
    y_inter = interpolate.interp1d(x, y)

    px_i = px - left_crop
    py = y[px_i]
    slope = get_local_slope(px_i, x, y)

    m = np.linspace(slope + SLOPE_INITIAL_TILT, 0, TILTING_STEPS, endpoint=False)
    current_slope = slope
    for mi in m:
        c = py - (mi * px_i)
        line = lambda x_: mi * x_ + c
        intersections = get_inversion_idx(line(x) - y_inter(x))
        if len(intersections) == INTERSECTION_NUMBER:
            current_slope = mi
        else:
            break

    final_line = lambda x_: current_slope * x_ + c
    intersections = get_inversion_idx(final_line(x) - y_inter(x))
    intersections_left_right = (intersections[0] + left_crop, intersections[-1] + left_crop)
    return intersections_left_right


def get_peak_boundaries_lake_drain(y: ndarray, px: float, left_crop: int) -> Tuple[int, int]:
    """
    Computes the left and right bounds of a peak, analysing the first derivative.
    First it interpolates the signal to be able to use it as a function.
    Starting at the top it "lowers" an horizontal line, as a lake would drain,
    as long as it has three intersection points with
    the interpolated functions.
    :param y: first derivative of the signal
    :param px: x of the maxima point
    :param left_crop: left cropping of the signal (for shifting)
    :return: left and right intersection
    """
    x = np.arange(0, len(y), 1, dtype=float)
    y_inter = interpolate.interp1d(x, y)

    px_i = px - left_crop
    py = y[px_i]
    m = 0

    c = np.linspace(py, 0, TILTING_STEPS, endpoint=False)

    current_height = py

    for ci in c:
        line = lambda x_: m * x_ + ci
        intersections = get_inversion_idx(line(x) - y_inter(x))
        if len(intersections) == INTERSECTION_NUMBER:
            current_height = c
        else:
            break

    final_line = lambda x_: m * x_ + current_height
    intersections = get_inversion_idx(final_line(x) - y_inter(x))
    intersections_left_right = (intersections[0] + left_crop, intersections[-1] + left_crop)
    return intersections_left_right


def get_inversion_idx(array: ndarray) -> List[int]:
    """
    Finds the points where the function changes sign
    :param array:
    :return:
    """
    a = np.sign(array)
    current_sign = a[0]
    idxs = []
    for i in range(len(a)):
        if current_sign * a[i] > 0:
            current_sign = a[i]
        elif current_sign * a[i] < 0:
            idxs.append(i)
            current_sign = a[i]
    return idxs


def get_local_slope(px_i: float, x: ndarray, y: ndarray) -> float:
    """
    Computes the local slope around a point
    :param px_i: x value of the point
    :param x: array of x values
    :param y: array of y values
    :return: the slope
    """

    p_neigh_x = x[px_i - LOCAL_POINTS:px_i + LOCAL_POINTS]
    p_neigh_y = y[px_i - LOCAL_POINTS:px_i + LOCAL_POINTS]
    slope, intercept, r, p, se = linregress(p_neigh_x, p_neigh_y)
    return slope


def popupmsg(msg, title):
    win = Tk()
    win.geometry("670x200")
    path = "sad_rat.jpg"
    img = ImageTk.PhotoImage(Image.open(path))
    win.title(title)
    label = tk.Label(win, text=msg)  # image=img)
    label.pack(fill="both", expand="yes")
    B1 = tk.Button(win, text="Okay! I will fix it later", command=win.destroy)
    B1.pack()
    win.mainloop()


def derrivate(data):  # input is Data_L and Data_R respectively
    """
    Calculating the derrivative of the kinematic data
    :param data: kinematic data (e.g. Data_L and Data_R)
    :return: derrivative of the kinematic data
    """
    derr = []
    for i in range(1, (len(data) - 1)):
        new_value = (data[i + 1] - data[i - 1]) / 2
        derr.append(new_value)
    return derr


def find_min(d_derrivate):
    """
    Finds the minimal values of the derrivative
    :param d_derrivate: derrivative
    :return: minima
    """
    minimum = []
    for i in range(0, len(d_derrivate)):
        if -0.0001 < d_derrivate[i] < 0.0001:
            minimum.append(i)
    return minimum


# find threshold value
def find_threshold(data):
    """
    Finds threshold based on mean of the minima of derrivative
    :param data: minima as calculated by function find_min
    :return: mean of the minimas (threshold for the ggait event_detection function) + 20 percent to account for trends
    etc. in data
    """
    th = statistics.median(data)
    return th + th * 0.2
