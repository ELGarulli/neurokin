from typing import List

import numpy as np
import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.core import FeatureExtraction
from neurokin.utils.kinematics.kinematics_extraction import linear_features

class LinearVelocity(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, fs: float, **kwargs):

        bodyparts_coordinates = df.columns.tolist()
        target_markers_coords = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if marker in coord]
        column_names = {bodypart: f"{bodypart}_linear_velocity" for bodypart in target_markers_coords}

        df_feat = df[target_markers_coords].apply(linear_features.compute_velocity, fs=fs)
        df_feat.rename(columns=column_names, inplace=True)

        return df_feat

class LinearAcceleration(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, fs: float, **kwargs):

        bodyparts_coordinates = df.columns.tolist()
        target_markers_coords = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if marker in coord]
        column_names = {bodypart: f"{bodypart}_linear_acceleration" for bodypart in target_markers_coords}

        df_feat = df[target_markers_coords].apply(linear_features.compute_acceleration, fs=fs)
        df_feat.rename(columns=column_names, inplace=True)

        return df_feat


class LinearSpeed(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, fs: float, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        feature_df_list = []
        for marker in target_bodyparts:
            coords = [coord for coord in bodyparts_coordinates if marker in coord]
            feat = linear_features.compute_speed(df[coords], fs=fs)
            df_feat = pd.DataFrame(feat, columns=[f"{marker}_linear_speed"])
            feature_df_list.append(df_feat)

        df_feat = pd.concat(feature_df_list, axis=1)
        return df_feat


class TangentialAcceleration(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, fs: float, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        feature_df_list = []
        for marker in target_bodyparts:
            coords = [coord for coord in bodyparts_coordinates if marker in coord]
            feat = linear_features.compute_tang_acceleration(df[coords], fs=fs)
            df_feat = pd.DataFrame(feat, columns=[f"{marker}_tang_acceleration"])
            feature_df_list.append(df_feat)

        df_feat = pd.concat(feature_df_list, axis=1)
        return df_feat