import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from neurokin.utils.kinematics.gait_params_basics import get_angle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict


# neurokin.utils.features.join_angles_dlc2kin.py
# mimics functions from dlc2kinematics.join_analysis.py
def compute_joint_angles(df: NDArray, joints_dict: Dict, dropnan: bool =False, smooth: bool =False,
                         filter_window: int=3, order: int=1, pcutoff: float =0.4) -> NDArray:
    """
    Computes the joint angles for the bodyparts.

    Parameters
    ----------
    df: Numpy array which is the output of software such as DeepLabCut. Assumes numpy array is already smoothed. If not, adjust the filter_window and order to smooth the array.

    joints_dict: Dictionary
        Keys of the dictionary specifies the joint angle and the corresponding values specify the bodyparts. e.g.
        joints_dict = {'R-Elbow': [1, 2, 3]}

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    smooth: boolean
        Optional. If you want to smooth the data with a svagol filter, you can set this to true, and then also add filter_window and order.

    filter_window: int
        Optional. If smooth=True,  window is set here, which needs to be a positive odd integer.

    order: int
        Optional. Only used if the optional argument `smooth` is set to True. Order of the polynomial to fit the data. The order must be less than the filter_window

    pcutoff: float
        Optional. Specifies the likelihood. All bodyparts with low `pcutoff` (i.e. < 0.4) are not used to compute the joint angles. It is only useful when computing joint angles from 2d data.

    Outputs
    -------
    joint_angles: numpy array of joint angles

    Example
    -------
    >>> joint_angles = neurokin.utils.kinematics.kinematics_extraction.angular_features.compute_joint_angles(df,joint_dict)

    """
    # does this need to be implemented?
    #flag, _ = auxiliaryfunctions.check_2d_or_3d(df)
    #if flag == "2d" and pcutoff:

    #    def filter_low_prob(cols, prob):
    #        mask = cols.iloc[:, 2] < prob
    #        cols.iloc[mask, :2] = np.nan
    #        return cols

    #    df = df.groupby("bodyparts", axis=1, group_keys=False).apply(filter_low_prob, prob=pcutoff)

    angles = np.zeros((len(joints_dict),df.shape[-1]))
    for n, entry in enumerate(joints_dict.items()):
        joint, bpts = entry
        print(f"Computing joint angles for {joint}")
        angles[n, :] = np.array([get_angle(df[bpts, :, i]) for i in range(df.shape[-1])])

    if dropnan:
        print("Dropping the indices where joint angle is nan")
        angles = angles[:, ~np.isnan(angles).any(axis=0)]

    if smooth:
        angles = savgol_filter(angles, filter_window, order, deriv=0, axis=-1)

    return angles


def compute_joint_velocity(joint_angle: NDArray, filter_window: int = 3, order: int = 1, dropnan: bool = False) -> NDArray:
    """
    Computes the joint angular velocities.

    Parameters
    ----------
    joint_angle: Numpy array of joint angles.

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    Outputs
    -------
    joint_vel: numpy array of joint angular velocity

    Example
    -------
    >>> joint_vel = neurokin.utils.kinematics.kinematics_extraction.angular_features.compute_joint_velocity(joint_angle)
    """
    #how to import?
    #try:
        #joint_angle = pd.read_hdf(joint_angle, "df_with_missing")
    #except:
    #    pass

    angular_vel = savgol_filter(joint_angle, filter_window, order, axis=-1, deriv=1)

    if dropnan:
        print("Dropping the indices where joint angular velocity is nan")
        angular_vel = angular_vel[:, ~np.isnan(angular_vel).any(axis=0)]

    return angular_vel


def compute_joint_acceleration(joint_angle: NDArray, filter_window: int = 3, order: int = 2, dropnan: bool = False) -> NDArray:
    """
    Computes the joint angular acceleration.

    Parameters
    ----------
    joint_angle: Numpy array of joint angles.

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer.

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window.

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    Outputs
    -------
    joint_acc: dataframe of joint angular acceleration.


    Example
    -------
    >>> joint_acc = neurokin.utils.kinematics.kinematics_extraction.angular_features.compute_joint_acceleration(joint_angle)
    """
    #how to import?
    #try:
        #joint_angle = pd.read_hdf(joint_angle, "df_with_missing")
    #except:
    #    pass

    angular_acc = savgol_filter(joint_angle, filter_window, order, axis=-1, deriv=2)

    if dropnan:
        print("Dropping the indices where joint angular acceleration is nan")
        angular_acc = angular_acc[:, ~np.isnan(angular_acc).any(axis=0)]

    return angular_acc


# neurokin.utils.features.correlations_dlc2kin.py
# mimics functions from dlc2kinematics.join_analysis.py
def compute_correlation(feature: NDArray, plot: bool = False, colormap: str = "viridis", keys: List[str] = None) -> NDArray:
    """
    Computes the correlation between the joint angles.

    Parameters
    ----------
    feature: Numpy array of joint anglular feature e.g. angular velocity. You can also pass the full path of joint anglular feature filename as a string.

    plot: Bool
        Optional. Plots the correlation.

    colormap: string
        Optional. The colormap associated with the matplotlib. Check here for range of colormap options https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    keys: List of strings
        Optional. List of joint names corresponding to rows of feature. If not provided and plot = True, will default to the index.

    Outputs
    -------
    corr: numpy array of correlation.


    Example
    -------
    >>> corr = neurokin.utils.kinematics.kinematics_extraction.angular_features.compute_correlation(joint_vel, plot=True, keys = ["R_hip", "R_knee", "L_hip", "L_knee"])
    """
    #how to import?
    #try:
        #feature = pd.read_hdf(feature, "df_with_missing")
    #except:
     #   pass

    correlation = np.corrcoef(feature)
    print(correlation.shape)
    if plot:
        if keys is None:
            keys = np.arange(0, correlation.shape[0], 1)
        im = plt.matshow(correlation, cmap=colormap)
        ax = plt.gca()
        plt.title("Correlation")
        plt.xticks(np.arange(0, len(keys), 1.0))
        plt.yticks(np.arange(0, len(keys), 1.0))
        ticks_labels = keys

        ax.set_xticklabels(ticks_labels, rotation=90)
        ax.set_yticklabels(ticks_labels)
        plt.gca().xaxis.tick_bottom()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)

        plt.colorbar(im, cax=cax)
        plt.clim(0, 1)
        plt.show()

    return correlation
