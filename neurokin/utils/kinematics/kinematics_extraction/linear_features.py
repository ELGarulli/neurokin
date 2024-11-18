from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter


def compute_speed(df: pd.DataFrame, fs: float, filter_window: int = 3, order: int = 1) -> NDArray:
    traj = df.apply(compute_velocity, args=(fs, ))
    speed = np.apply_along_axis(np.linalg.norm, 1, traj, None, 0)
    return speed


def compute_velocity(df: pd.DataFrame, fs: float) -> NDArray:
    df = df.values
    return np.gradient(df, (1 / fs))


def compute_acceleration(df: pd.DataFrame, fs: float) -> NDArray:
    velocity = compute_velocity(df, fs)
    return np.gradient(velocity, (1 / fs))


def compute_tang_acceleration(df: pd.DataFrame, fs: float) -> NDArray:
    speed = compute_speed(df, fs)
    return np.gradient(speed, (1 / fs))


def _compute_velocity(df: NDArray, bodyparts: List[int], filter_window: int = 3, order: int = 1) -> NDArray:
    """
    Computes the velocity of bodyparts in the input dataframe.

    Parameters
    ----------
    df: numpy.array
        Assumes the numpy array is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodypart indeces to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    vel: numpy array of velocity for the bodypart

    Example
    -------
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_velocity(df,bodyparts=[1,3])

    To smooth all the bodyparts in the dataframe, use
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_velocity(df,bodyparts=['all'])

    """
    return smooth_trajectory(df, bodyparts, filter_window, order, deriv=1)


def _compute_speed(df: NDArray, bodyparts: List[int], filter_window: int = 3, order: int = 1) -> NDArray:
    """
    Computes the speed of bodyparts in the input dataframe.

    Parameters
    ----------
    df: numpy.array
        Assumes the numpy array is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodypart indeces to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    speed: numpy array of speed for the bodyparts

    Example
    -------
    >>> speed = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_speed(df,bodyparts=[1,3])

    To smooth all the bodyparts in the dataframe, use
    >>> speed = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_speed(df, bodyparts=['all'])
    """
    traj = smooth_trajectory(df, bodyparts, filter_window, order, deriv=1)
    # coords = traj.columns.get_level_values("coords") != "likelihood"
    # prob = traj.loc[:, ~coords]

    # def _calc_norm(cols):
    # return np.sqrt(np.sum(cols ** 2, axis=1))

    # groups = (
    #    ["individuals", "bodyparts"]
    #    if "individuals" in df.columns.names
    #    else "bodyparts"
    # )
    # vel = traj.loc[:, coords].groupby(level=groups, axis=1).apply(_calc_norm)
    vel = np.apply_along_axis(np.linalg.norm, 0, traj, None, 0)  # What is the expected shape?

    # scorer = df.columns.get_level_values("scorer").unique().to_list()
    # try:
    #    levels = vel.columns.levels
    # except AttributeError:
    #    levels = [vel.columns.values]
    # vel.columns = pd.MultiIndex.from_product(
    #    [scorer] + levels + [["speed"]],
    #    names=["scorer"] + vel.columns.names + ["coords"],
    # )
    # return vel.join(prob)
    return vel


def _compute_acceleration(df: NDArray, bodyparts: List[int], filter_window: int = 3, order: int = 2) -> NDArray:
    """
    Computes the acceleration of bodyparts in the input dataframe.

    Parameters
    ----------
    df: numpy.array
        Assumes the numpy array is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodypart indeces to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    vel: numpy array of velocity for the bodypart

    Example
    -------
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_acceleration(df,bodyparts=[1,3])

    To smooth all the bodyparts in the dataframe, use
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.compute_acceleration(df,bodyparts=['all'])

    """
    return smooth_trajectory(df, bodyparts, filter_window, order, deriv=2)


# neurokin.kinematic_data.py
# mimics functions from dlc2kinematics.preprocess
def smooth_trajectory(df: NDArray, bodyparts: List[int], filter_window: int = 3, order: int = 1,
                      deriv: int = 0) -> NDArray:
    """
    Smooths the input data which is a numpy array generated by DeepLabCut as a result of analyzing a video.

    Parameters
    ----------
    df: numpy.array

    bodyparts: List
        List of bodypart indeces to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        The length of filter window which needs to be a positive odd integer

    order: int
        Order of the polynomial to fit the data. The order must be less than the filter_window

    deriv: int
        Optional. Computes the derivative. If order=1, it computes the velocity on the smoothed data, if order=2 it computes the acceleration on the smoothed data.

    Outputs
    -------
    df: smoothed numpy array

    Example
    -------
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.smooth_trajectory(df,bodyparts=[1,3],window_length=11,order=3)

    To smooth all the bodyparts in the dataframe, use
    >>> df_smooth = neurokin.utils.kinematics.kinematics_extraction.linear_features.smooth_trajectory(df,bodyparts=['all'],window_length=11,order=3)

    """
    df_new = np.copy(df)
    if bodyparts[0] == "all":
        to_smooth = np.ones(df.shape[0], dtype=bool)
    else:
        to_smooth = np.arange(df.shape[0]) == bodyparts
    df_new = savgol_filter(df_new[to_smooth, :, :], filter_window, order, deriv, axis=-1)

    return df_new
