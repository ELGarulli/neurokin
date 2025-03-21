from functools import partial
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter

from neurokin.utils.features_extraction import feature_extraction
from neurokin.utils.kinematics import binning
from neurokin.utils.helper import load_config
from neurokin.utils.kinematics import kinematics_processing, import_export, event_detection

class KinematicDataRun:
    """
    This class represents the kinematics data recorded in a single run.
    """

    def __init__(self, path, configpath):
        self.path = path

        self.config = load_config.read_config(configpath)

        self.trial_roi_start: int
        self.trial_roi_end: int
        self.fs: float
        self.condition: str

        self.markers_df = pd.DataFrame()
        self.gait_param = pd.DataFrame()
        self.stepwise_gait_features = pd.DataFrame()
        self.features_df: pd.MultiIndex = None
        self.binned_df: pd.DataFrame = None

        self.left_mtp_lift: ArrayLike = None
        self.left_mtp_land: ArrayLike = None
        self.left_mtp_max: ArrayLike = None
        self.right_mtp_lift: ArrayLike = None
        self.right_mtp_land: ArrayLike = None
        self.right_mtp_max: ArrayLike = None
        self.bodyparts: ArrayLike = None
        self.scorer: str = "scorer"

    def load_kinematics(self,
                        correct_shift: bool = False,
                        correct_tilt: bool = False,
                        to_shift: ArrayLike = None,
                        to_tilt: ArrayLike = None,
                        shift_reference_marker: str = "",
                        tilt_reference_marker: str = "",
                        source: str = None,
                        fs: float = None):
        """
        Loads the kinematics from a c3d file into a dataframe with timeframes as rows and markers as columns

        :param correct_shift: bool should there be a correction in the shift of one of the axis?
        :param correct_tilt: bool should there be a correction in the tilt (linear trend) of one of the axis?
        :param to_shift: which columns to perform the shift on if correct_shift is true
        :param to_tilt: which columns to perform the tilt on if correct_shift is true
        :param shift_reference_marker: which marker to use as a reference trajectory to compute the shift
        :param tilt_reference_marker: which marker to use as a reference trajectory to compute the tilt
        :param source: defines which source of kinematics data to load from, either c3d or dlc
        :return:
        """

        if source.lower() == "c3d":
            self.trial_roi_start, self.trial_roi_end, self.fs, self.markers_df = import_export.import_c3d(self.path)
        elif source.lower() == "dlc":
            self.markers_df = import_export.import_dlc_df(self.path)
            self.convert_DLC_like_to_df()
            self.fs = fs
        else:
            raise ValueError(f"The source value {source} is not yet implemented. Currently supported sources are 'c3d' "
                             f"and 'dlc' (DeepLabCut)")

        if correct_shift:

            if shift_reference_marker not in self.markers_df.columns.get_level_values("bodyparts").tolist():
                raise ValueError("The shift reference marker " + shift_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 self.markers_df.columns.tolist())

            if not set(to_shift).issubset(self.markers_df.columns.get_level_values("bodyparts").tolist()):
                raise ValueError("Some or all columns to shift are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_shift)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.shift_correct(self.markers_df, shift_reference_marker, to_shift)

        if correct_tilt:

            if tilt_reference_marker not in self.markers_df.columns.get_level_values("bodyparts").tolist():
                raise ValueError("The tilt reference marker " + tilt_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            if not set(to_tilt).issubset(self.markers_df.columns.get_level_values("bodyparts").tolist()):
                raise ValueError("Some or all columns to tilt are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_tilt)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.tilt_correct(self.markers_df, tilt_reference_marker, to_tilt)

        return

    def filter_marker_df(self, func=None, **kwargs):
        """
        Filters markers dataframe in place using a given function or defaults on a savgol filter with window_length 4 and
        polyorder 2. Pass the arguments to change the defaults.
        :param func: user-given filter function. Must be used as an applied function on a dataset
        :param kwargs: user-given parameters for the savgol filter, or the custom filter function.
        :return:
        """
        if not func:
            func = partial(savgol_filter, window_length=4, polyorder=2)
        original_shape = self.markers_df.shape
        filtered_df = self.markers_df.apply(lambda col: func(col, **kwargs))
        if filtered_df.shape != original_shape:
            raise ValueError(
                f"Shape mismatch, the filter function has to return the same shape as the input. "
                f"Expected {original_shape} returned {filtered_df.shape}")
        self.markers_df = filtered_df

    def convert_DLC_like_to_df(self, multiindex_df=None):
        """
        Converts a DeepLabCut-like dataframe (MultiIndex) to a simple pandas dataframe. Saves the scorer identity and
        the bodyparts in the corresponding class attributes.

        :param smooth: whether to smooth the coordinates or not
        :param filter_window: how big should the filter be
        :param order: order of the polynomial to fit the data
        :return:
        """
        if not multiindex_df:
            multiindex_df = self.markers_df
        bodyparts = multiindex_df.columns.get_level_values("bodyparts").unique().to_list()
        scorers = np.unique(multiindex_df.columns.get_level_values(0))
        scorer = [i for i in scorers if "Unnamed" not in i][0]
        selected_df = multiindex_df.loc[:, multiindex_df.columns.get_level_values("scorer") == scorer]
        selected_df.columns = ["_".join(a[1:]) for a in selected_df.columns.to_flat_index()]
        selected_df.reset_index(inplace=True, drop=True)
        self.markers_df = selected_df
        self.bodyparts = bodyparts
        self.scorer = scorer
        return

    def compute_gait_cycles_bounds(self, left_marker, right_marker, axis="z"):
        """
        Computes the lifting and landing frames of both feet using a left and a right marker, respectively.
        To increase robustness of the cycle estimation it first low-passes the signal.

        :param left_marker: reference marker for the left foot, typically the left mtp
        :param right_marker: reference marker for the right foot, typically the right mtp
        :param recording_fs: sample frequency of the recording, used for low-passing.
        :param axis: axis to use to trace the movement and set the cycle bounds
        :return:
        """

        if left_marker not in self.markers_df.columns.get_level_values("bodyparts").unique():
            raise ValueError("The left reference marker " + left_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.get_level_values("bodyparts").unique()))

        if right_marker not in self.markers_df.columns.get_level_values("bodyparts").unique():
            raise ValueError("The right reference marker " + right_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.get_level_values("bodyparts").unique()))

        self.left_mtp_lift, self.left_mtp_land, self.left_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[self.scorer][left_marker][axis], self.fs)
        self.right_mtp_lift, self.right_mtp_land, self.right_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[self.scorer][right_marker][axis], self.fs)

        return

    def plot_step_partition(self, step_left, step_right, ax_l, ax_r):
        """
        Plots the step partition of a gait trace (ideally toe or foot trace). Fetches the class attributes so the
        steps computation has to happen before

        :param step_left: name of the left trace body part
        :param step_right: name of the right trace body part
        :param ax_l: ax to plot the left trace on
        :param ax_r: ax to plot the right trace on
        :return: axes
        """
        step_trace_l = self.markers_df[self.scorer][step_left]["z"]
        ax_l.plot(step_trace_l)
        ax_l.vlines(self.left_mtp_lift, min(step_trace_l), max(step_trace_l), colors="green")
        ax_l.vlines(self.left_mtp_land, min(step_trace_l), max(step_trace_l), colors="red")
        ax_l.set_title("Left side")

        step_trace_r = self.markers_df[self.scorer][step_right]["z"]
        ax_r.plot(step_trace_r)
        ax_r.vlines(self.right_mtp_lift, min(step_trace_r), max(step_trace_r), colors="green")
        ax_r.vlines(self.right_mtp_land, min(step_trace_r), max(step_trace_r), colors="red")
        ax_r.set_title("Right side")

        ax_l.set_xlim(0, len(step_trace_l))
        ax_r.set_xlim(0, len(step_trace_r))

        return ax_l, ax_r

    def print_step_partition(self, step_left, step_right, output_folder="./"):
        """
        Creates and saves the step partition plot. See plot_step_partition method for details.

        :param step_left: name of the left trace body part
        :param step_right: name of the right trace body part
        :param output_folder: where to save the figure
        :return:
        """
        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=2.0)
        filename = output_folder + self.path.split("/")[-1] + "_steps_partition.png"
        axs[0], axs[1] = self.plot_step_partition(step_left, step_right, axs[0], axs[1])
        plt.savefig(filename, facecolor="white")
        plt.close()

    def extract_features(self, get_binned=True, custom_feats=None):
        """
        Computes features on the markers dataframe based on the config file. If get_binned is True, it also computes
        the binned features as defined in the config file.

        :return:
        """
        features = self.config["features"]
        skeleton = self.config["skeleton"]
        binning = self.config["binning"]
        new_features, new_binned_features = feature_extraction.extract_features(features=features,
                                                                                bodyparts=self.bodyparts,
                                                                                skeleton=skeleton,
                                                                                markers_df=self.markers_df,
                                                                                get_binned=get_binned,
                                                                                bin_params=binning,
                                                                                custom_feats=custom_feats)

        if self.features_df is not None:
            self.features_df = pd.concat((self.features_df, new_features), axis=1)
        else:
            self.features_df = new_features

        if self.binned_df is not None:
            self.binned_df = pd.concat((self.binned_df, new_binned_features), axis=1)
        else:
            self.binned_df = new_binned_features

    def get_trace_height(self, marker, axis="z", window=50, overlap=25):
        """
        Computes height traces of a given marker

        :param marker: which marker to consider
        :param axis: x, y, or z, default z
        :param window: window to compute the binning on
        :param overlap: how much should 2 subsequent windows overlab by
        :return:
        """
        test = binning.get_step_height_on_bins(self.markers_df,
                                               window=window,
                                               overlap=overlap,
                                               marker=marker,
                                               axis=axis)
        return test

    def get_step_fwd_movement_on_bins(self, marker, axis="y", window=50, overlap=25):
        """
        Computes forward movement traces of a given marker

        :param marker: which marker to consider
        :param axis: x, y, or z, default z
        :param window: window to compute the binning on
        :param overlap: how much should 2 subsequent windows overlab by
        :return:
        """
        test = binning.get_step_fwd_movement_on_bins(self.markers_df,
                                                     window=window,
                                                     overlap=overlap,
                                                     marker=marker,
                                                     axis=axis)
        return test

    def gait_param_to_csv(self, output_folder="./"):
        """
        Writes the gait_param dataframe to a csv file with the name [INPUT_FILENAME]+_gait_param.csv

        :return:
        """
        self.gait_param.to_csv(output_folder + self.path.split("/")[-1].replace(".c3d", "_gait_param.csv"))
        return

    def stepwise_gait_features_to_csv(self, output_folder="./"):
        """
        Writes the gait_param dataframe to a csv file with the name [INPUT_FILENAME]+_gait_param.csv

        :return:
        """
        self.stepwise_gait_features.to_csv(
            output_folder + self.path.split("/")[-1].replace(".c3d", "stepwise_feature.csv"))
        return

    def get_angles_features(self, features_df, **kwargs):
        left_df, right_df = self.split_in_unilateral_df(**kwargs)
        left_features = pd.DataFrame()
        right_features = pd.DataFrame()

        left_features = kinematics_processing.get_angle_features(left_df, self.left_mtp_land,
                                                                 features_df)  # TODO update features df with all angle features

        self.stepwise_gait_features = left_features.join(right_features)
