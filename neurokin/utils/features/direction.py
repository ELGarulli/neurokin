import pandas as pd
from typing import List, Dict, Any
from neurokin.utils.features.core import FeatureExtraction
import numpy as np


class Direction(FeatureExtraction):

    """
    Computes the direction of the subject in the maze
    Input:  df with positon data (i.e. DLC output), source_marker_ids
    Output: df with direction information (Bool) for the subject (scorer, subject, direction)
            (i.e. True if towards open, False if towards closed)
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "front_marker": ["Snout"],
            "back_marker": ["TailBase"],
            "axis": "x",
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "front_marker": [str],
            "back_marker": [str],
            "axis": [str],
        }
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # check if direction is already determined
        if ("scorer", "subject", "direction") in marker_df.columns:
            pass
        else:
            # create new col in df for direction
            marker_df.loc[:, ("scorer", "subject", "direction")] = False
            marker_df.loc[
                marker_df[("scorer", params["front_marker"], params["axis"])]
                > marker_df[("scorer", params["back_marker"], params["axis"])],
                ("scorer", "subject", "direction"),
            ] = True
            # only keep the direction column
            direction_df = (
                self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="direction",
                    data=marker_df.loc[:, ("scorer", "subject", "direction")],
                )
            )


            return direction_df
