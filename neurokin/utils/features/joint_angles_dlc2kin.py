import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any

from neurokin.utils.kinematics.csv_import_export import convert_singleindex_to_multiindex_df
from neurokin.utils.features.core import FeatureExtraction, DefaultParams


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
            filter_window=params["window_size"], save=False
        )
        # reshape df to multiindex df
        bp = list(source_marker_ids.keys())[0]
        df_col = convert_singleindex_to_multiindex_df(scorer='scorer', bodypart=bp, axis='angle', data=df_joint_angles)

        # self._assert_valid_output(output_df=df_joint_reshaped, marker_df=marker_df)
        return df_col


class AngularVelocityDLC(FeatureExtraction):
    """
    Computes the velocity of angles
    Input: df with joint angle data, source_marker_ids: List of markers for which speed should be computed
    Output: df with angular velocity for input markers
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

        # filter df for specific columns, raise error if angles not calculated yet
        # angels_df = marker_df.loc[:, ('scorer', source_marker_ids, 'angle')]

        joint = list(source_marker_ids.keys())[0]
        try:
            angels_df = self._copy_filtered_columns_of_df(df_to_filter=marker_df,
                                                      marker_id_filter=joint,
                                                      coords_filter='angle')
        except ValueError:
            print('No column "angles" found for joint', joint)

        df_angular_momentum = dlc2kinematics.compute_joint_velocity(
            joint_angle=angels_df,
            filter_window=params["window_size"], save=False)
        # self._assert_valid_output(output_df=df_angular_momentum, marker_df=marker_df)

        # reshape df to multiindex df
        df_angular_momentum = convert_singleindex_to_multiindex_df(scorer='scorer',
                                                                   bodypart=joint,
                                                                   axis='angle',
                                                                   data=df_angular_momentum)
        return df_angular_momentum
