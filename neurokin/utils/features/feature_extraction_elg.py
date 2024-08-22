from importlib import import_module

import pandas as pd

from neurokin.constants.features_extraction import FEATURES_EXTRACTION_MODULE


def get_extractor_obj(feature_name):
    module_, feature_class = feature_name.rsplit(".", maxsplit=1)
    module_ = FEATURES_EXTRACTION_MODULE + module_
    m = import_module(module_)
    feature_extract_class = getattr(m, feature_class)
    return feature_extract_class()


def extract_features(features, bodyparts, skeleton, markers_df):
    extracted_features = []
    for feature_name, params in features.items():
        params = params if params else {}
        extractor_obj = get_extractor_obj(feature_name)
        extraction_target = extractor_obj.extraction_target

        if extraction_target == "markers":
            target_bodyparts = params.get("marker_ids", bodyparts)

        elif extraction_target == "joints":
            target_joints = params.get("markers_ids", skeleton["angles"][extraction_target])
            target_bodyparts = [{joint: skeleton["angles"][extraction_target][joint]} for joint in target_joints]

        elif extraction_target == "multiple_markers":
            target_bodyparts = params.get("marker_ids", bodyparts)

        for bodypart in target_bodyparts:
            feature = extractor_obj.run_feat_extraction(bodypart=bodypart, markers_df=markers_df, params=params)
            extracted_features.append(feature)

    return pd.DataFrame(extracted_features)
