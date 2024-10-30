import numpy as np
from neurokin.utils.features_extraction.core_elg import FeatureExtraction
from typeguard import typechecked
import pandas as pd
from typing import List


class LinearSpeed(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, window_size: int, markers_id: List):
        target = markers_id if markers_id else target_bodyparts
        bodyparts_coordinates = df.columns.tolist()
        target_markers_coords = [coord for marker in target for coord in bodyparts_coordinates if marker in coord]
        df_feat = df[target_markers_coords].apply(np.diff).apply(abs)
        return df_feat
