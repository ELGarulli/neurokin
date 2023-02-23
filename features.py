import dlc2kinematics
from typing import List, Dict, Any

from .core import FeatureExtraction, DefaultParams
from utils import DLC_conversion # soon to be implemented

class Feature_velocity(FeatureExtraction):
    """
    Computes the velocity of bodyparts seperated in the three dimensions -> 3 values (x,y,z)
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which velocity should be computed
    Output: df with velocity data for input markers
    """

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {'window_size': 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {'window_size': [int]}
        return default_types

    def _run_feature_extraction(self, source_marker_ids):
        # scorer will be added as index
        filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
                                                        marker_id_filter=source_marker_ids)
        df_velocity = dlc2kinematics.compute_velocity(df=filtered_df, filter_window = self.params['window_size'])
        self._assert_valid_output(output_df=df_velocity)

        return df_velocity # scorer would be removed


class Feature_speed(FeatureExtraction):
    """
    Computes the speed of bodyparts in a 3D space -> 1 value
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which speed should be computed
    Output: df with speed data for input markers
    """
    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {'window_size': 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {'window_size': [int]}
        return default_types


    def _run_feature_extraction(self, source_marker_ids):

        filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
                                                        marker_id_filter=source_marker_ids)
        df_speed = dlc2kinematics.compute_speed(df=filtered_df, filter_window = self.params['window_size'])
        self._assert_valid_output(output_df=df_speed)
        return df_speed

class Feature_acceleration(FeatureExtraction):
    """
    Computes the acceleration of bodyparts
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which velocity should be computed
    Output: df with acceleration data for input markers
    """
    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {'window_size': 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {'window_size': [int]}
        return default_types

    def _run_feature_extraction(self, source_marker_ids):

        filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
                                                        marker_id_filter=source_marker_ids)
        df_acceleration = dlc2kinematics.compute_acceleration(df=filtered_df, filter_window = self.params['window_size'])
        self._assert_valid_output(output_df=df_acceleration)
        return df_acceleration

class Feature_joint_angles(FeatureExtraction):
    """
    Computes the angles of joints
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which speed should be computed
    Output: df with joint angle data for input markers
    """
    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {'window_size': 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {'window_size': [int]}
        return default_types


    def _run_feature_extraction(self, source_marker_ids):

        filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
                                                        marker_id_filter=source_marker_ids)
        df_speed = dlc2kinematics.compute_joint_angles(df=filtered_df, joints_dict= self.joints, filter_window = self.params['window_size'])
        self._assert_valid_output(output_df=df_speed)
        return df_speed

class Feature_angular_velocity(FeatureExtraction):
    """
    Computes the velocity of angles
    Input: df with joint angle data, source_marker_ids: List of markers for which speed should be computed
    Output: df with angular velocity for input markers
    """

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {'window_size': 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {'window_size': [int]}
        return default_types

    def _run_feature_extraction(self, source_marker_ids):
        filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
                                                        marker_id_filter=source_marker_ids)
        df_speed = dlc2kinematics.compute_joint_velocity(joint_angle=filtered_df, joints_dict=self.joints,
                                                       filter_window=self.params['window_size'])
        self._assert_valid_output(output_df=df_speed)
        return df_speed


