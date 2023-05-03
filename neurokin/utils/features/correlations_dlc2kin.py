import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any

from neurokin.utils.features.core import FeatureExtraction, DefaultParams


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

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:

        # filter df for specific columns, raise error if angles not calculated yet
        # params defining properties
        filtered_df = self._copy_filtered_columns_of_df(
            df_to_filter=marker_df, marker_id_filter=source_marker_ids
        )


        df_correlation = dlc2kinematics.compute_correlation(filtered_df)

        names = marker_df.columns.names
        scorer = marker_df.columns.get_level_values("scorer")[0]
        columns = ["angle" for i in range(len(df_correlation.columns.to_list()))]
        df_correlation_reshaped = df_correlation.copy()
        df_correlation_reshaped.columns = pd.MultiIndex.from_product(
            [[scorer], source_marker_ids.keys(), columns],
            names=names)

        self._assert_valid_output(output_df=df_correlation_reshaped, marker_df=marker_df)
        return df_correlation_reshaped