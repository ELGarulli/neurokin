from importlib import import_module
from typing import List
import pandas as pd

from neurokin.constants.features_extraction import FEATURES_EXTRACTION_MODULE


def get_extractor_obj(feature_name):
    module_, feature_class = feature_name.rsplit(".", maxsplit=1)
    module_ = FEATURES_EXTRACTION_MODULE + module_
    m = import_module(module_)
    feature_extract_class = getattr(m, feature_class)
    return feature_extract_class()


def extract_features(features, bodyparts, skeleton, markers_df, get_binned, bin_params, custom_feats):
    extracted_features = []
    binned_features = []
    binned_df = None
    for feature_name, params in features.items():
        params = params if params else {}
        params["custom_features"] = custom_feats
        extractor_obj = get_extractor_obj(feature_name)
        extraction_target = extractor_obj.extraction_target

        if extraction_target == "markers":
            target_bodyparts = bodyparts if len(params["marker_ids"])==0 else params["marker_ids"]

        elif extraction_target == "joints":
            target_joints = skeleton[extraction_target].keys() if len(params["joint_ids"])==0 else params["joint_ids"]
            target_bodyparts = {joint: skeleton[extraction_target][joint] for joint in target_joints}

        elif extraction_target == "misc":
            target_bodyparts = params.get("misc_ids", bodyparts)

        else:
            raise ValueError(f"{extraction_target} is not a valid extraction target."
                             f"Please use: markers, joints or multiple_markers")

        feature = extractor_obj.run_feat_extraction(df=markers_df, target_bodyparts=target_bodyparts, **params)
        extracted_features.append(pd.DataFrame(feature).reset_index(drop=True))
        if get_binned:
            try:
                binning_strategy = params["binning_strategy"]
            except KeyError:
                continue
            binned_features.append(bin_feature(feature,
                                               binning_strategies=binning_strategy,
                                               window=bin_params["window_size"],
                                               overlap=bin_params["overlap"]).reset_index(drop=True))
            binned_df = pd.concat(binned_features, axis=1)
    feats_df = pd.concat(extracted_features, axis=1)
    return feats_df, binned_df


def bin_feature(feature, binning_strategies: List[str], window, overlap):
    step = window - overlap
    binned_features = []
    if step <= 0:
        raise ValueError(f"The overlap should be lower than the window. Got overlap: {overlap} and window: {window}.")
    for strategy in binning_strategies:
        if strategy.lower().strip(" ") == "mean":
            binned = feature.rolling(window=window, step=step).mean().add_suffix("_mean")
        elif strategy.lower().strip(" ") == "median":
            binned = feature.rolling(window=window, step=step).median().add_suffix("_median")
        elif strategy.lower().strip(" ") == "min":
            binned = feature.rolling(window=window, step=step).min().add_suffix("_min")
        elif strategy.lower().strip(" ") == "max":
            binned = feature.rolling(window=window, step=step).max().add_suffix("_max")
        elif strategy.lower().strip(" ") == "sum":
            binned = feature.rolling(window=window, step=step).sum().add_suffix("_sum")
        elif strategy.lower().strip(" ") == "std":
            binned = feature.rolling(window=window, step=step).std().add_suffix("_std")
        else:
            raise ValueError(f"The chose binning strategy: {strategy} is not available. Please choose between:"
                             f"mean, median, min, max, sum or std")
        binned_features.append(binned)
    binned_features_df = pd.concat(binned_features, axis=1)
    return binned_features_df
