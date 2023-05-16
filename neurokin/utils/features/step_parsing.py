import pandas as pd
import numpy as np

from numpy import ndarray
from scipy import signal

from typing import List, Dict, Any, Tuple

from neurokin.utils.features.core import FeatureExtraction, DefaultParams


def _get_peaks(y, params):
    # peak detection is based on Code from Elisa Garulli https://github.com/WengerLab/neurokin/blob/d077050bc8760abfe24928b53ad6383e96f59902/utils/kinematics/event_detection.py
    """
    Returns the left and right bounds of the gait cycle, corresponding to the toe lift off and the heel strike.
    :param y: trace of the toe in the z coordinate
    :return: left and right bounds
    """
    y = _lowpass_array(y, params['step_filter_freq'], params['fps'])

    max_x, _ = signal.find_peaks(y, prominence=params['prominence'])

    if len(max_x) > 2:
        avg_distance = abs(int(_median_distance(max_x) / 2))

        lb = []
        rb = []

        for p in max_x:
            left = p - avg_distance if p - avg_distance > 0 else 0
            right = p + avg_distance if p + avg_distance < len(y) else len(y)
            bounds = _get_peak_boundaries_scipy(
                y=y[left:right], px=p, left_crop=left, RELATIVE_HEIGHT=params['relative_height']
            )
            lb.append(bounds[0])
            rb.append(bounds[1])

        lb = np.asarray(lb)
        rb = np.asarray(rb)
        return lb, rb, max_x
    else:
        return [], [], []

def _lowpass_array(array, critical_freq, fs):
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


def _get_peak_boundaries_scipy(y: ndarray, px: float, left_crop: int, RELATIVE_HEIGHT: float
                               ) -> Tuple[int, int]:
    peaks = np.asarray([px - left_crop])
    peak_pro = signal.peak_prominences(y, peaks)
    peaks_width = signal.peak_widths(
        y, peaks, rel_height=RELATIVE_HEIGHT, prominence_data=peak_pro
    )
    intersections = peaks_width[-2:]
    try:
        left = intersections[0] + left_crop
    except:
        left = left_crop

    try:
        right = intersections[-1] + left_crop
    except:
        right = len(y) + left_crop

    return [int(left), int(right)]


def _median_distance(a: ndarray) -> ndarray:
    """
    Gets median distance between peaks
    :param a:
    :return: median
    """
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return np.median(distances)


class StepParsing(FeatureExtraction):
    """
    Computes Gait cycle and marks step start, peak and end, as well as swing and stance phase in two columns in df
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers that are used for step detection
    Output: df with additional columns for gait phase and relevant step information (start, peak, end)
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "smoothing": False,
            "smoothing_window_size": 40,
            "relative_height": 0.9,
            "step_filter_freq": 4,
            "prominence": 5,
            "fps": 80,
            "min_x_diff": 0.1,
            "marker_ids": ["ForePawRight", "ForePawLeft", "HindPawRight", "HindPawLeft"],
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "smoothing": [bool],
            "smoothing_window_size": [int],
            "relative_height": [float],
            "step_filter_freq": [int],
            "prominence": [int],
            "fps": [int],
            "min_x_diff": [float],
            "marker_ids": [list],
        }
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # Check whether step cycle is already calculated
        if ("scorer", source_marker_ids, "gait_cycle_phase") in marker_df.columns:
            pass
        else:
            # Check whether speed is calculated for all markers critical for immobility
            if 'speed' in marker_df.loc[:, ('scorer', source_marker_ids)].columns.all():

                speed_df = marker_df.loc[:, ('scorer', source_marker_ids, ['speed'])]
                speed = speed_df['scorer'][source_marker_ids]['speed']
                x_df = marker_df.loc[:, ('scorer', source_marker_ids, 'x')]

                # check whether smoothing is desired
                if params["smoothing"]:
                    pass

                # get peak indices
                start, end, peak = _get_peaks(y = speed, params = params)

                # check whether it is actually a step
                # calculate the difference in x between start and end
                x_movement_bouts = []
                for startidx, endidx in zip(start, end):
                    x_diff = abs(startidx-endidx)
                    if x_diff > params['min_x_diff']:
                        bout_range = np.arange(startidx, endidx + 1)
                        x_movement_bouts.append(bout_range)
                x_movement_indices = np.concatenate(x_movement_bouts)

                # get indices of correct start, peak, and end of gait cycle bouts
                correct_start = np.intersect1d(start, x_movement_indices)
                correct_peak = np.intersect1d(peak, x_movement_indices)
                correct_end = np.intersect1d(end, x_movement_indices)

                # create array of indices that include all indices of a bout
                bouts = []
                for startidx, endidx in zip(correct_start, correct_end):
                    bout_range = np.arange(startidx, endidx + 1)
                    bouts.append(bout_range)
                bout_indices = np.concatenate(bouts)

                # write gait phase into df
                speed_df.loc[:, ('scorer',source_marker_ids,'gait_cycle_phase')] = 'stance'
                speed_df.iloc[bout_indices, speed_df.columns.get_loc(('scorer', source_marker_ids, 'gait_cycle_phase'))] = 'swing'
                # write gait phase markers into df
                speed_df.loc[:, ('scorer', source_marker_ids, 'gait_cycle_phase_markers')] = np.nan
                speed_df.iloc[correct_start, speed_df.columns.get_loc(('scorer', source_marker_ids, 'gait_cycle_phase_markers'))] = 'start'
                speed_df.iloc[correct_peak, speed_df.columns.get_loc(('scorer', source_marker_ids, 'gait_cycle_phase_markers'))] = 'peak'
                speed_df.iloc[correct_end, speed_df.columns.get_loc(('scorer', source_marker_ids, 'gait_cycle_phase_markers'))] = 'end'

                # convert dfs into multiindex df
                gait_phases = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart=source_marker_ids,
                    axis="gait_cycle_phase",
                    data=speed_df.loc[:, ('scorer', source_marker_ids, 'gait_cycle_phase')],
                )
                gait_phase_marker = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart=source_marker_ids,
                    axis="gait_cycle_phase_markers",
                    data=speed_df.loc[:, ('scorer', source_marker_ids, 'gait_cycle_phase_markers')],
                )


                final_gait_cycle_df = pd.concat(
                    [
                        gait_phases,
                        gait_phase_marker
                    ],
                    axis=1,
                )
                return final_gait_cycle_df

            else:
                raise ValueError(
                    "Be sure to extract speed for all bodyparts that you want to use for step detection!"
                )