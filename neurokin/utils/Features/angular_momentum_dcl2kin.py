import dlc2kinematics
from typing import List, Dict, Any

from .core import FeatureExtraction, DefaultParams


class Feature_angular_velocity(FeatureExtraction):
    """
    Computes the velocity of angles
    Input: df with joint angle data, source_marker_ids: List of markers for which speed should be computed
    Output: df with angular velocity for input markers
    """

    @property
    def input_type(self):
        return 'joints'

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int]}
        return default_types

    def _run_feature_extraction(self, source_marker_ids):
        df_angular_momentum = dlc2kinematics.compute_joint_velocity(
            joint_angle=self.marker_df,
            joints_dict=source_marker_ids,
            filter_window=self.params["window_size"],
        )
        self._assert_valid_output(output_df=df_angular_momentum)
        return df_angular_momentum
