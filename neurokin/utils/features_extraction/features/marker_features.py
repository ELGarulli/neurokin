import numpy as np
from neurokin.utils.features_extraction.core_elg import FeatureExtraction
from typeguard import typechecked


class LinearSpeed(FeatureExtraction):
    extraction_target = "markers"

    @typechecked


    def compute_feature(self, df, target_bodyparts, window_size):
        #df_feat = self.df[self.target_bodyparts].apply(np.diff)
        return "df_feat"
