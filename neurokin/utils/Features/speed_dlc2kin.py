import dlc2kinematics
from typing import List, Dict, Any

from .core import FeatureExtraction, DefaultParams


class Feature_speed(FeatureExtraction):
    """
    Computes the speed of bodyparts in a 3D space -> 1 value
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which speed should be computed
    Output: df with speed data for input markers
    """

    @property
    def input_type(self):
        return 'markers'

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int]}
        return default_types

    def _run_feature_extraction(self, source_marker_ids):

        df_speed = dlc2kinematics.compute_speed(
            df=self.marker_df,
            bodyparts=source_marker_ids,
            filter_window=self.params["window_size"],
        )
        self._assert_valid_output(output_df=df_speed)
        return df_speed
