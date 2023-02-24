import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any

from .core import FeatureExtraction, DefaultParams


class JointAnglesDLC(FeatureExtraction):
    """
    Computes the angles of joints
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which speed should be computed
    Output: df with joint angle data for input markers
    """

    @property
    def input_type(self) -> str:
        return "joints"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int]}
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:

        df_joint_angles = dlc2kinematics.compute_joint_angles(
            df=marker_df,
            joints_dict=source_marker_ids,
            filter_window=params["window_size"],
        )
        self._assert_valid_output(output_df=df_joint_angles, marker_df=marker_df)
        return df_joint_angles


class AngularVelocityDLC(FeatureExtraction):
    """
    Computes the velocity of angles
    Input: df with joint angle data, source_marker_ids: List of markers for which speed should be computed
    Output: df with angular velocity for input markers
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int]}
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        df_angular_momentum = dlc2kinematics.compute_joint_velocity(
            joint_angle=marker_df,
            filter_window=params["window_size"],)
        self._assert_valid_output(output_df=df_angular_momentum, marker_df=marker_df)
        return df_angular_momentum
