# from const import (DATAFILE, PATHNAME, SUBJECT)
from utils.kinematics.event_detection import get_toe_lift_landing
import numpy as np


def load_raw_kinematics(eng, gait_file, pathname):
    h = eng.minEx_0(gait_file, pathname, nargout=1)
    h = eng.minEx_1(h, nargout=1)
    return h

def get_gait_cycle_bounds(h, data_name: str, s):
    """
    event detection function using detection based on line from peaks, intersection at 2 points
    (see get_left_right_bound for explanation).

    :param h: handles
    :param data_right_name: eg "Data_R"
    :param data_left_name: eg "Data_L"
    :param s: boolean value, adjusted by clicking yes or no in popup window, if no (not happy with event detection)
    defaulting to threshold method
    :return: thresholds, bounds_array_hs, bounds_array_to, uneven_r, uneven_l, left_toe_off, left_heel_strike, \
           right_toe_off, right_heel_strike, data_current_r, data_current_l, y
    """


    raw_data = h[data_name]
    raw_data = np.array(raw_data)
    y = [i[0] for i in raw_data]
    y = np.array(y)
    bounds_array_toe_off, bounds_array_heel_strike, maxima_points = get_toe_lift_landing(y=y)

    gait_cycle_dict = {
        "toe_off": bounds_array_toe_off,
        "heel_strike": bounds_array_heel_strike,
        "maxima_points": maxima_points,
        "raw_data": y}

    return gait_cycle_dict
