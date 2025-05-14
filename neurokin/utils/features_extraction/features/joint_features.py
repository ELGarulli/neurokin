from typing import Dict
import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.commons import (compute_angle,
                                                        compute_angle_correlation,
                                                        compute_angle_acceleration,
                                                        compute_angle_velocity,
                                                        compute_angle_phase)
from neurokin.utils.features_extraction.core import FeatureExtraction


class Angle(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = compute_angle(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class AngleVelocity(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = compute_angle_velocity(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_velocity"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class AngleAcceleration(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = compute_angle_acceleration(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_acceleration"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class AngleCorrelation(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = compute_angle_correlation(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_correlation"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class AnglePhase(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = compute_angle_phase(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_phase"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class CustomJointFeatures(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        try:
            feature_names = kwargs.get("feature_names")
        except KeyError:
            raise KeyError(f"No name found for the Custom Join Function. "
                           f"Please add valid feature_names in the config file.")

        for name in feature_names:
            try:
                func = kwargs.get("custom_features")[name]
            except KeyError:
                raise KeyError(f"No function found with name: {name}, please provide a valid name and function "
                               f"name when calling extract_features")
            except TypeError:
                raise TypeError(f"Custom Join Functions are called in the config file but not passed while calling. "
                                f"Please either delete the custom features in the config file or pass them upon calling"
                                f"feature extraction.")
            for joint, bodyparts in target_bodyparts.items():
                target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                         marker in coord]
                feat = func(df[target_markers_coords].values)
                df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_{name}"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat
