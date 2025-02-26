from typing import List
import numpy as np
import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.core import FeatureExtraction


class Height(FeatureExtraction):
    """
    Computes Height of selected markers and one selected coordinate (x, y, z). By default it sets the lowest point to 0
    so all other points are referenced to it.
    """
    extraction_target = "misc"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, coord: str, normalize_min: bool = True, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        target_markers = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if
                                 marker in coord]
        target_markers_coords = [marker for marker in target_markers if coord in marker]

        for bodypart in target_markers_coords:
            feat = df[bodypart].values
            if normalize_min:
                feat = feat - np.nanmin(feat)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{bodypart}_height"]))
        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat



class FwdMovement(FeatureExtraction):
    """
    Computes Forward Movement of selected markers and one selected coordinate (x, y, z), by returning the first
    derivative. By default it sets the lowest point to 0 so all other points are referenced to it.
    """
    extraction_target = "misc"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, coord: str, normalize_min: bool = True, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        target_markers = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if
                                 marker in coord]
        target_markers_coords = [marker for marker in target_markers if coord in marker]

        for bodypart in target_markers_coords:
            feat = df[bodypart].values
            if normalize_min:
                feat = feat - np.nanmin(feat)
            feat = np.diff(feat)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{bodypart}_fwd_movement"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat