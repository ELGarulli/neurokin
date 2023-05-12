import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any

from neurokin.utils.features.core import FeatureExtraction, DefaultParams


class JointAnglesDLC(FeatureExtraction):
    """
    Computes the angles of joints
    Input: df with positon data (i.e. DLC output), source_marker_ids: dictionary of markers for which angles should be computed
    Output: df with joint angle data
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
        try:
            angles = list(source_marker_ids.values())[0]
            df_angles = marker_df.loc[:, ("scorer", angles, ["x", "y", "z"])]
            df_joint_angles = dlc2kinematics.compute_joint_angles(
                df=df_angles,
                joints_dict=source_marker_ids,
                filter_window=params["window_size"],
                save=False,
            )
            # reshape df to multiindex df
            bp = list(source_marker_ids.keys())[0]
            df_col = self.convert_singleindex_to_multiindex_df(
                scorer="scorer", bodypart=bp, axis="angle", data=df_joint_angles
            )

            # self._assert_valid_output(output_df=df_joint_reshaped, marker_df=marker_df)
            return df_col

        except ValueError:
            print('WARNING: Joint angle cannot be calculated for ', angles,
                  '. Check the coverage of the DLC model.')
            return None


class AngularVelocityDLC(FeatureExtraction):
    """
    Computes the velocity of angles
    Input: df with joint angle data, source_marker_ids: List of joints for which speed should be computed
    Output: df with angular velocity for input markers
    """

    @property
    def input_type(self) -> str:
        return "joints"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3, "fps": 80}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int], "fps": [int]}
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:

        # Check if angle is calculated. Print warning if not
        try:
            joint = list(source_marker_ids.keys())[0]
            angels_df = self._copy_filtered_columns_of_df(
                df_to_filter=marker_df, marker_id_filter=joint, coords_filter="angle"
            )


            df_angular_momentum = dlc2kinematics.compute_joint_velocity(
                joint_angle=angels_df, filter_window=params["window_size"], save=False
            )
            # convert from px/frame to px/s
            df_angular_momentum = df_angular_momentum * params["fps"]
            # reshape df to multiindex df
            df_angular_momentum = self.convert_singleindex_to_multiindex_df(
                scorer="scorer",
                bodypart=joint,
                axis="angular_momentum",
                data=df_angular_momentum,
            )
            return df_angular_momentum
        except KeyError:
            print('WARNING: No column "angles" found for joint', joint,
                  '. Angular velocity cannot be calculated.')
            return None

