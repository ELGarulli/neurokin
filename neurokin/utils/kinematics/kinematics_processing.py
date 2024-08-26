import numpy as np
import pandas as pd
from scipy import signal
from neurokin.utils.kinematics.gait_params_basics import get_angle, get_phase_at_max_amplitude

#TESTME with mock df
def get_marker_coordinates_names(df_columns_names, markers):
    """
    Returns the names of the columns that contain the name of the marker, to retrieve the 2 or 3 coordinates.
    E.g. lknee, will could return lknee_x, lknee_y, lknee_z.

    :param df_columns_names: dataframe column names with all the markers
    :param markers: markers of interest
    :return:
    """
    abc = []
    for i in range(len(markers)):
        point = [x for x in df_columns_names if markers[i] in x]
        abc.append(point)
    abc.sort()  # courtesy of me dreaming code. makes xyz order assumption more likely, still an assumption.
    return tuple(abc)

#TESTME with mock df
def get_marker_coordinate_values(df, marker_column_names, frame):
    """
    Given a dataframe, a list of column names referring to x, y (and z) of the same marker, and a frame number,
    it returns the corresponding values.

    :param df: dataframe
    :param marker_column_names: sets of markers names
    :param frame: frame number
    :return: sets of coordinates values
    """
    coordinates = []
    for i in range(len(marker_column_names)):
        coordinates.append(df[marker_column_names[i]][frame])
    return coordinates


#TESTME with mock df
def tilt_correct(df, reference_marker, columns_to_correct):
    """
    If the runway is not perfectly aligned there can be a linear trend in one of the axis.
    This function computes the linear trend from a reference marker and applies it to all the columns passed.
    E.g. use left mtp z axis to compute the trend, then subtract the trend from all columns representing the z axis
    of a marker.

    """

    trend = signal.detrend(df[reference_marker]) - df[reference_marker]
    df_tilt_corrected = df.apply(lambda x: x.add(trend, axis=0) if x.name in columns_to_correct else x)
    return df_tilt_corrected

#TESTME with mock df
def shift_correct(df, reference_marker, columns_to_correct):
    #TODO compliance with DLC?
    """
    If the origin is not set to the beginning of the runway (e.g. set to the center) one of the axis will have negative
    values. This functions shifts all the columns to be corrected by the minimum value of the reference marker.
    The reference marker should be the one farther in the back.

    """

    shift = abs(min(df[reference_marker])) if min(df[reference_marker]) < 0 else 0
    df_shift_corrected = df.apply(lambda x: x.add(shift, axis=0) if x.name in columns_to_correct else x)
    return df_shift_corrected


def check_correct_columns_extraction(actual, expected, side):
    if actual == expected:
        return
    else:
        raise ValueError("Warning: the number of selected columns for the " + side + " side [" + str(actual) +
                         "] does not match the \n"
                         "expected " + str(expected) +
                         ". Please check if there are ambiguity in the column names.")

#TESTME with mock df for the the 3 correct cases
def get_unilateral_df(df, side="", name_starts_with=False, name_ends_with=False,
                      column_names=None, expected_columns_number=None):
    if column_names is None:
        column_names = []
    if name_starts_with:
        df_side = df.loc[:, df.columns.str.startswith(side)]
    elif name_ends_with:
        df_side = df.loc[:, df.columns.str.endswith(side)]
    elif column_names:
        df_side = df.loc[:, column_names]
    else:
        print("WARNING: no columns selected for side " + side + ". Please check if this is expected behaviour. \n"
                                                                "If not, you should set the side and either starts "
                                                                "or ends with, alternatively pass column_names.")
        return
    if expected_columns_number:
        check_correct_columns_extraction(len(df_side.columns.tolist()), expected_columns_number, side)

    return df_side


def get_angle_features(df, breakpoints, features_df):
    # df_feature = pd.DataFrame(columns=df.columns)
    for column in df.features_df:
        gait_param = np.asarray(df[column])
        steps_gait_param = np.split(gait_param, breakpoints)[:-1]
        steps_feat = [max(i) for i in steps_gait_param]
        features_df[column] = steps_feat

    bodyparts = []
    features = []
    for bodypart in bodyparts:
        steps_gait_param = np.split(gait_param, breakpoints)[:-1]
        for feature in features:

            if feature == "max_angle":
                for step in range(len(steps_gait_param)):
                    max_ = np.max(steps_gait_param[step])
                    features_df[feature][bodypart].iloc[step] = max_

            if feature == "min_angle":
                for step in range(len(steps_gait_param)):
                    min_ = np.min(steps_gait_param[step])
                    features_df[feature][bodypart].iloc[step] = min_

            if feature == "phase_max_amplitude":
                for step in range(len(steps_gait_param)):
                    ph_at_max_amplitude = get_phase_at_max_amplitude(steps_gait_param[step])
                    features_df[feature][bodypart].iloc[step] = ph_at_max_amplitude

    return


def create_empty_features_df(self, bodyparts, features):
    # TODO check shape passed: not matching
    dataFrame = None
    steps_number = max([len(self.right_mtp_land), len(self.left_mtp_land)])
    a = np.full((steps_number), np.nan)
    for bodypart in bodyparts:
        pdindex = pd.MultiIndex.from_product(
            [features, [bodypart]],
            names=["feature", "bodypart"])
        frame = pd.DataFrame(a, columns=pdindex, index=range(0, steps_number))
        dataFrame = pd.concat([frame, dataFrame], axis=1)
    return dataFrame