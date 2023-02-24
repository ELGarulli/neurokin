from importlib import import_module
from neurokin.constants.features_extraction import FEATURES_EXTRACTION_MODULE
import pandas as pd


def extract_features(features, bodyparts, skeleton, markers_df):
    markers_and_features_df = markers_df.copy()
    for feature_name, params in features.items():

        module_, feature_class = feature_name.rsplit(".", maxsplit=1)
        module_ = FEATURES_EXTRACTION_MODULE + module_
        m = import_module(module_)
        feature_extract_class = getattr(m, feature_class)
        extractor_obj = feature_extract_class()
        input_type = extractor_obj.input_type

        if input_type == "markers":
            target_bodyparts = params.get("marker_ids", bodyparts)

        elif input_type == "joints":
            target_joints = params.get("marker_ids", skeleton["angles"][input_type])
            target_bodyparts = [{joint: skeleton["angles"][input_type][joint]} for joint in target_joints]

        params.pop("marker_ids", None)

        extracted_features = []

        for bodypart in target_bodyparts:
            feature = extractor_obj.extract_features(source_marker_ids=bodypart,
                                                     marker_df=markers_and_features_df,
                                                     params=params)
            extracted_features.append(feature)

        new_features = pd.concat(extracted_features, axis=1)
        markers_and_features_df = pd.concat((markers_and_features_df, new_features), axis=1)

    return new_features
