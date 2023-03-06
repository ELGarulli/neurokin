from typing import Tuple, List, Dict, Optional, Union, Any
import pandas as pd
from .core import FeatureExtraction, DefaultParams


class SampleFeatureExtractionStrategy(FeatureExtraction):
    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 5, "aggregation_method": "mean"}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int], "aggregation_method": [str]}
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        filtered_df = self._copy_filtered_columns_of_df(
            df_to_filter=marker_df, marker_id_filter=source_marker_ids
        )
        if params["aggregation_method"] == "mean":
            filtered_df = filtered_df.rolling(params["window_size"]).mean()
        elif params["aggregation_method"] == "sum":
            filtered_df = filtered_df.rolling(params["window_size"]).sum()
        else:
            raise NotImplementedError(
                "The SampleFeatureExtractionStrategy does not yet include an implementation "
                f'to aggregate the data using the requested "{params["aggregation_method"]}" '
                'method. The currently available options are: "mean" and "sum".'
            )
        filtered_df = self._rename_columns_on_selected_idx_level(
            df=filtered_df, suffix=f'_sliding_{params["aggregation_method"]}'
        )
        return filtered_df
