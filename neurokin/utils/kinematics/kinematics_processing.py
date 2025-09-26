import numpy as np
import pandas as pd
from scipy import signal



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

