import numpy as np
from neurokin.utils.features_extraction.core import FeatureExtraction
from typeguard import typechecked
import pandas as pd
from typing import List


class LinearSpeed(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, window_size: int, **kwargs):

        bodyparts_coordinates = df.columns.tolist()
        target_markers_coords = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if marker in coord]
        column_names = {bodypart: f"{bodypart}_linear_speed" for bodypart in target_markers_coords}

        df_feat = df[target_markers_coords].apply(np.diff).apply(abs)
        df_feat.rename(columns=column_names, inplace=True)

        return df_feat

