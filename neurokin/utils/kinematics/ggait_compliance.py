from neurokin.utils.kinematics.event_detection import get_toe_lift_landing
import numpy as np

def load_raw_kinematics(eng, gait_file, pathname):
    h = eng.minEx_0(gait_file, pathname, nargout=1)
    h = eng.minEx_1(h, nargout=1)
    return h


def get_gait_cycle_bounds(h, data_name: str, fs=200):
    """
    event detection function using detection based on line from peaks, intersection at 2 points
    (see get_left_right_bound for explanation).
    :param h: handles
    :param data_name: eg "Data_R"
    :param fs: sampling frequency of kinematic data
    :return: thresholds, bounds_array_hs, bounds_array_to, uneven_r, uneven_l, left_toe_off, left_heel_strike, \
           right_toe_off, right_heel_strike, data_current_r, data_current_l, y
    """

    raw_data = h[data_name]
    raw_data = np.array(raw_data)
    y = [i[0] for i in raw_data]
    y = np.array(y)
    bounds_array_toe_off, bounds_array_heel_strike, maxima_points = get_toe_lift_landing(y=y, recording_fs=fs)

    gait_cycle_dict = {
        "toe_off": np.asarray(bounds_array_toe_off),
        "heel_strike": np.asarray(bounds_array_heel_strike),
        "maxima_points": np.asarray(maxima_points)
       }


    return gait_cycle_dict