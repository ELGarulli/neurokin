import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any
from neurokin.utils.features.core import FeatureExtraction, DefaultParams


class VelocityDLC(FeatureExtraction):
    """
    Computes the velocity of bodyparts seperated in the three dimensions -> 3 values (x,y,z)
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which velocity should be computed
    Output: df with velocity data for input markers
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

        df_velocity = dlc2kinematics.compute_velocity(
            df=marker_df,
            bodyparts=[source_marker_ids],
            filter_window=params["window_size"]
        )
        self._assert_valid_output(output_df=df_velocity, marker_df=marker_df)

        return df_velocity


class SpeedDLC(FeatureExtraction):
    """
    Computes the speed of bodyparts in a 3D space -> 1 value
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which speed should be computed
    Output: df with speed data for input markers
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
        df_speed = dlc2kinematics.compute_speed(
            df=marker_df,
            bodyparts=[source_marker_ids],
            filter_window=params["window_size"]
        )
        self._assert_valid_output(output_df=df_speed, marker_df=marker_df)
        return df_speed


class AccelerationDLC(FeatureExtraction):
    """
    Computes the acceleration of bodyparts
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which velocity should be computed
    Output: df with acceleration data for input markers
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

        df_acceleration = dlc2kinematics.compute_acceleration(
            df=marker_df,
            bodyparts=[source_marker_ids],
            filter_window=params["window_size"]
        )
        self._assert_valid_output(output_df=df_acceleration, marker_df=marker_df)
        return df_acceleration
