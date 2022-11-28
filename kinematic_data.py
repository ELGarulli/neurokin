import numpy as np
from numpy.typing import ArrayLike
from utils.kinematics import c3d_import_export, event_detection, kinematics_processing
from utils.helper import load_config
import pandas as pd
from matplotlib import pyplot as plt


class KinematicDataRun:
    """
    This class represents the kinematics data recorded in a single run.
    """

    def __init__(self, path, configpath):
        self.path = path

        self.config = load_config.read_config(configpath)

        self.gait_cycles_start: int
        self.gait_cycles_end: int
        self.fs: float
        self.condition: str

        self.markers_df = pd.DataFrame()
        self.gait_param = pd.DataFrame()
        self.stepwise_gait_features = pd.DataFrame()

        self.left_mtp_lift: ArrayLike = None
        self.left_mtp_land: ArrayLike = None
        self.left_mtp_max: ArrayLike = None
        self.right_mtp_lift: ArrayLike = None
        self.right_mtp_land: ArrayLike = None
        self.right_mtp_max: ArrayLike = None

    def load_kinematics(self,
                        correct_shift: bool = False,
                        correct_tilt: bool = False,
                        to_shift: ArrayLike = None,
                        to_tilt: ArrayLike = None,
                        shift_reference_marker: str = "",
                        tilt_reference_marker: str = ""):
        """
        Loads the kinematics from a c3d file into a dataframe with timeframes as rows and markers as columns
        :param correct_shift: bool should there be a correction in the shift of one of the axis?
        :param correct_tilt: bool should there be a correction in the tilt (linear trend) of one of the axis?
        :param to_shift: which columns to perform the shift on if correct_shift is true
        :param to_tilt: which columns to perform the tilt on if correct_shift is true
        :param shift_reference_marker: which marker to use as a reference trajectory to compute the shift
        :param tilt_reference_marker: which marker to use as a reference trajectory to compute the tilt
        :return:
        """

        self.gait_cycles_start, self.gait_cycles_end, self.fs, self.markers_df = c3d_import_export.import_c3d(self.path)

        if correct_shift:

            if shift_reference_marker not in self.markers_df.columns.tolist():
                raise ValueError("The shift reference marker " + shift_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 self.markers_df.columns.tolist())

            if not set(to_shift).issubset(self.markers_df.columns.tolist()):
                raise ValueError("Some or all columns to shift are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_shift)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.shift_correct(self.markers_df, shift_reference_marker, to_shift)

        if correct_tilt:

            if tilt_reference_marker not in self.markers_df.columns.tolist():
                raise ValueError("The tilt reference marker " + tilt_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            if not set(to_tilt).issubset(self.markers_df.columns.tolist()):
                raise ValueError("Some or all columns to tilt are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_tilt)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.tilt_correct(self.markers_df, tilt_reference_marker, to_tilt)
        return


    def compute_gait_cycles_bounds(self, left_marker, right_marker):
        """
        Computes the lifting and landing frames of both feet using a left and a right marker, respectively.
        To increase robustness of the cycle estimation it first low-passes the signal.
        :param left_marker: reference marker for the left foot, typically the left mtp
        :param right_marker: reference marker for the left foot, typically the right mtp
        :param recording_fs: sample frequency of the recording, used for low-passing.
        :return:
        """

        if left_marker not in self.markers_df.columns.tolist():
            raise ValueError("The left reference marker " + left_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.tolist()))

        if right_marker not in self.markers_df.columns.tolist():
            raise ValueError("The right reference marker " + right_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.tolist()))

        self.left_mtp_lift, self.left_mtp_land, self.left_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[left_marker], self.fs)
        self.right_mtp_lift, self.right_mtp_land, self.right_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[right_marker], self.fs)

        return

    def print_step_partition(self, step_left, step_right, output_folder="./"):
        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=2.0)
        filename = output_folder + self.path.split("/")[-1] + "_steps_partition.png"
        step_trace_l = self.markers_df[step_left]
        axs[0].plot(step_trace_l)
        axs[0].vlines(self.left_mtp_lift, min(step_trace_l), max(step_trace_l), colors="green")
        axs[0].vlines(self.left_mtp_land, min(step_trace_l), max(step_trace_l), colors="red")
        axs[0].set_title("Left side")

        step_trace_r = self.markers_df[step_right]
        axs[1].plot(step_trace_r)
        axs[1].vlines(self.right_mtp_lift, min(step_trace_r), max(step_trace_r), colors="green")
        axs[1].vlines(self.right_mtp_land, min(step_trace_r), max(step_trace_r), colors="red")
        axs[1].set_title("Right side")
        plt.savefig(filename, facecolor="white")
        plt.close()

    def compute_angles_joints(self):
        """
        It refers to the joints listed in the config under angles > joints to set a corresponding column in the
        gait_param dataset. It should be able to support both 3d and 2d spaces.
        :return:
        """
        for key, value in self.config["angles"]["joints"].items():
            names = kinematics_processing.get_marker_coordinates_names(self.markers_df.columns.tolist(), value)
            angle = []
            for frame in range(len(self.markers_df)):
                coordinates_3d = []
                for name in names:
                    values = kinematics_processing.get_marker_coordinate_values(self.markers_df, name, frame)
                    coordinates_3d.append(values)
                coordinates_3d = np.asarray(coordinates_3d)
                angle.append(kinematics_processing.compute_angle(coordinates_3d))
            parameter = pd.Series(angle)
            self.gait_param[key] = parameter
        return

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
        self.stepwise_gait_features.to_csv(output_folder + self.path.split("/")[-1].replace(".c3d", "stepwise_feature.csv"))
        return

    def create_empty_features_df(self, bodyparts, features):
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

    def get_angles_features(self, features_df, **kwargs):
        left_df, right_df = self.split_in_unilateral_df(**kwargs)
        left_features = pd.DataFrame()
        right_features = pd.DataFrame()

        left_features = kinematics_processing.get_angle_features(left_df, self.left_mtp_land, features_df) #TODO update features df with all angle features

        self.stepwise_gait_features = left_features.join(right_features)

    def split_in_unilateral_df(self, left_side="", right_side="",
                               name_starts_with=False, name_ends_with=False,
                               expected_columns_number=None,
                               left_columns=None, right_columns=None):
        #self.stepwise_gait_features = 0
        left_df = pd.DataFrame()
        right_df = pd.DataFrame()
        if left_side:
            left_df = kinematics_processing.get_unilateral_df(df=self.gait_param, side=left_side,
                                                              name_starts_with=name_starts_with,
                                                              name_ends_with=name_ends_with,
                                                              column_names=left_columns,
                                                              expected_columns_number=expected_columns_number)


        if right_side:
            right_df = kinematics_processing.get_unilateral_df(df=self.gait_param, side=right_side,
                                                               name_starts_with=name_starts_with,
                                                               name_ends_with=name_ends_with,
                                                               column_names=right_columns,
                                                               expected_columns_number=expected_columns_number)


        return left_df, right_df
