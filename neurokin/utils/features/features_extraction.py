from importlib import import_module
from neurokin.constants.features_extraction import FEATURES_EXTRACTION_MODULE
import pandas as pd
from typing import Dict, List, Any, Optional, Union


def extract_features(
    features: Dict, bodyparts: List, skeleton: Dict, markers_df: pd.DataFrame
) -> pd.DataFrame:
    markers_and_features_df = markers_df.copy()
    for feature_name, params in features.items():
        params = {} if params is None else params
        module_, feature_class = feature_name.rsplit(".", maxsplit=1)
        module_ = FEATURES_EXTRACTION_MODULE + module_
        m = import_module(module_)
        feature_extract_class = getattr(m, feature_class)
        extractor_obj = feature_extract_class()
        input_type = extractor_obj.input_type

        if input_type == "markers":  # single marker -> loop over
            target_bodyparts = params.get("marker_ids", bodyparts)

        elif input_type == "joints":  # ref in skeleton
            target_joints = params.get("marker_ids", skeleton["angles"][input_type])
            target_bodyparts = [
                {joint: skeleton["angles"][input_type][joint]}
                for joint in target_joints
            ]
        elif input_type == "distance":
            target_distance = params.get("marker_ids", skeleton["distances"])
            target_bodyparts = [
                {distance: skeleton["distances"][distance]}
                for distance in target_distance
            ]

            pass

        elif input_type == "multiple_markers":
            target_bodyparts = [params.get("marker_ids", bodyparts)]

        params.pop("marker_ids", None)

        extracted_features = []

        for bodypart in target_bodyparts:
            feature = extractor_obj.extract_features(
                source_marker_ids=bodypart,
                marker_df=markers_and_features_df,
                params=params,
            )
            if feature is not None:
                extracted_features.append(feature)
        # for some features (e.g. threshold_based_classification,
        # it´s possible that a feature is None for all markers
        # in this case, extracted_features is empty and cannot be concatenated
        # -> markers_and_features_df remains unchanged and is just returned
        if extracted_features:
            new_features = pd.concat(extracted_features, axis=1)
            markers_and_features_df = pd.concat(
                (markers_and_features_df, new_features), axis=1
            )

    return markers_and_features_df
