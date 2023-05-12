import dlc2kinematics
import pandas as pd
from typing import List, Dict, Any
from neurokin.utils.features.core import FeatureExtraction
import numpy as np


class Distance(FeatureExtraction):

    """
    Computes the distance between two markers of the df
    Input: df with positon data (i.e. DLC output), source_marker_ids: Dict
    Output: df with acceleration data for input markers
    """

    @property
    def input_type(self) -> str:
        return "distance"

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
        # get the name of the new column
        distance = list(source_marker_ids.keys())[0]
        # get the two markers for which the distance should be computed
        markers = list(source_marker_ids.values())[0]
        df_distance = self._copy_filtered_columns_of_df(
            df_to_filter=marker_df, marker_id_filter=markers
        )

        # get the values of the two markers
        point1 = df_distance.loc[:, ("scorer", markers[0], ["x", "y", "z"])].values
        point2 = df_distance.loc[:, ("scorer", markers[1], ["x", "y", "z"])].values

        # Calculate the Euclidean distance between point1 and point2 at each time point
        distances = np.sqrt(((point1 - point2) ** 2).sum(axis=1))
        df_temp = pd.DataFrame(distances)
        # Add the distances as a new column to the original DataFrame
        distance_df = self.convert_singleindex_to_multiindex_df(
            scorer="scorer", bodypart=distance, axis="distance", data=df_temp
        )
        return distance_df

        df_distance = self._rename_output_of_extraction_methods(
            df=df_distance, bodypart=source_marker_ids, suffix="_distance"
        )
        return df_distance
