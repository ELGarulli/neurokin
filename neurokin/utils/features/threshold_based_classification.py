import pandas as pd
from typing import List, Dict, Any

from neurokin.utils.features.core import FeatureExtraction, DefaultParams

class BodypartImmobility(FeatureExtraction):
    """
       Classifies bodyparts as immobile if the bodyparts speed is below a speed threshold
       Input: df with positon data (i.e. DLC output), source_marker_ids: List of m should be computed
       Output: df with joint angle data for input markers
       """


    @property
    def input_type(self) -> str:
        return "feature"


    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"immobility_threshold": 3}
        return default_values


    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"immobility_threshold": [int]}
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
        for marker_id in source_marker_ids:
            filtered_df.loc[:, (marker_id, 'immobility')] = False
            #function that
            filtered_df.loc[filtered_df[(marker_id, 'rolling_speed_px_per_s')]
                            < params["immobility_threshold"], (marker_id, 'immobility')] = True

