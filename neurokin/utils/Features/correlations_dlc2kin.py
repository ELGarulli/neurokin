import dlc2kinematics
from typing import List, Dict, Any

from .core import FeatureExtraction, DefaultParams


class CorrelationDLC(FeatureExtraction):
    """
    Computes the correlation of features
    Input: markers_df, source_marker_ids: List of features to be correlated
    Output: df with correlations for features
    """

    @property
    def input_type(self):
        return "correlations"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {}
        return default_types

    def _run_feature_extraction(self, marker_df, source_marker_ids):

        filtered_df = self._copy_filtered_columns_of_df(
            df_to_filter=marker_df, marker_id_filter=[source_marker_ids]
        )
        df_correlation = dlc2kinematics.compute_correlation(filtered_df)

        self._assert_valid_output(output_df=df_correlation, marker_df=marker_df)
        return df_correlation
