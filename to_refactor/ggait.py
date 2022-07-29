# from const import (DATAFILE, PATHNAME, SUBJECT)
from utils.kinematics.event_detection import *
import numpy as np
from scipy.signal import find_peaks
import matlab.engine


def events_detected(h, data_right, data_left, s):
    """
    event detection function using detection based on line from peaks, intersection at 2 points
    (see get_left_right_bound for explanation). If this does not provide couls results, event detection using
    automatically detected threshold and ggait event_detector function is used.

    :param h: handles
    :param data_right: "Data_R"
    :param data_left: "Data_L"
    :param s: boolean value, adjusted by clicking yes or no in popup window, if no (not happy with event detection)
    defaulting to threshold method
    :return: thresholds, bounds_array_hs, bounds_array_to, uneven_r, uneven_l, left_toe_off, left_heel_strike, \
           right_toe_off, right_heel_strike, data_current_r, data_current_l, y
    """
    bounds_array_to = []
    bounds_array_hs = []

    data_current_r = []
    data_current_l = []

    uneven_r = 0
    uneven_l = 0

    left_toe_off = 0
    left_heel_strike = 0

    right_toe_off = 0
    right_heel_strike = 0

    thresholds = []
    maximas = {}

    for string in [data_right, data_left]:
        data_current = h[string]
        data_current = np.array(data_current)
        y = [i[0] for i in data_current]
        y = np.array(y)
        max_x, _ = find_peaks(y, prominence=1)
        maximas[string] = max_x
        y_g = np.gradient(y, 1)
        avg_distance = abs(int(median_distance(max_x) / 2))
        lb = []
        rb = []

        for p in max_x:
            left = p - avg_distance if p - avg_distance > 0 else 0
            right = p + avg_distance if p + avg_distance < len(y_g) else len(y_g)
            bounds = get_left_right_bound(y=y_g[left:right], px=p, left_crop=left)
            lb.append(bounds[0])
            rb.append(bounds[1])

        bounds_array_to.append(lb)
        bounds_array_hs.append(rb)

        # plt.plot(y, color="black")
        # plt.vlines(max_x, ymin=-1, ymax=2, color="blue")
        # plt.vlines(lb, ymin=-1, ymax=2, color="green")
        # plt.vlines(rb, ymin=-1, ymax=2, color="red")
        # plt.show()
        method_2 = "yes"
        if not s:
            method_2 = "no"
            if string == "Data_R":
                uneven_r = 0
            elif string == "Data_L":
                uneven_l = 0

        """
        if len(np.unique(lb+rb)) < 2*len(max_x) and method_2 == "yes":
            data_current = h[string]
            data_current = np.array(data_current)
            data_current = [float(x) for x in list(data_current)]
            #data_current = remove_trend(data_current)
            #data_current = [float(x) for x in list(data_current)]
            #MsgBox = tkinter.messagebox.askyesno('Should the Data be detrended', 'Should the Data be detrended')
            #if MsgBox == 'yes':

            first = derrivate(data_current)
            second = derrivate(first)
            first_min = find_min(first)
            second_min = find_min(second)
            minima_first = []
            for j in first_min:
                minima_first.append(data_current[j])

            minima_second = []
            for k in second_min:
                minima_second.append(data_current[k])

            threshold_first = find_threshold(minima_first)
            threshold_second = find_threshold(minima_second)
            thresholds.append(threshold_first)
            if string == "Data_R":
                right_toe_off = thresholds[0]
                uneven_r = 1
                data_current_r = data_current
            else:
                left_toe_off = thresholds[0]
                uneven_l = 1
                data_current_l = data_current

            popupmsg("msg", "The events for " + string + " could not be detected. The default threshold based "
                                                         "method was used! ")
            """

        ### EVERYTHING BELOW IS ONLY RELEVANT IF I USE THE
        #   if yes == 1:
        #       new_array = []
        #       length = len(data_current)
        #      for i in range(0, length):
        #           new_array = np.append(new_array, data_current[i])
        #           new_array = np.array(new_array)
        #           y, idx = integratation(new_array)
        #           axis = np.linspace(0, 1, length)
        #       uneven_r = 0
        #       uneven_l = 0
        #       temp_to = idx[::2]
        #       temp_hs = idx[1::2]
        #       right_heel_strike = matlab.double(temp_hs)
        #       right_toe_off = matlab.double(temp_to)

    if uneven_l == 0:
        left_toe_off = matlab.double(bounds_array_to[1])
        left_heel_strike = matlab.double(bounds_array_hs[1])

    if uneven_r == 0:
        right_toe_off = matlab.double(bounds_array_to[0])
        right_heel_strike = matlab.double(bounds_array_hs[0])

    param_dict = {'thresholds': thresholds,
                  'bounds_array_hs': bounds_array_hs,
                  'bounds_array_to': bounds_array_to,
                  'uneven_r': uneven_r,
                  'uneven_l': uneven_l,
                  'left_toe_off': left_toe_off,
                  'left_heel_strike': left_heel_strike,
                  'right_toe_off': right_toe_off,
                  'right_heel_strike': right_heel_strike,
                  'data_current_r': data_current_r,
                  'data_current_l': data_current_l,
                  'y': y,
                  'maximas': maximas}

    return param_dict
