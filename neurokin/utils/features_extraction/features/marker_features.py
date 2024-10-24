import numpy as np
from neurokin.utils.features_extraction.core_elg import FeatureExtraction
from typeguard import typechecked

class LinearSpeed(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def __init__(self, df, target_bodyparts):
        self.df = df
        self.target_bodyparts = target_bodyparts

    def compute_feature(self):
        df_feat = self.df[self.target_bodyparts].apply(np.diff)
        return df_feat
