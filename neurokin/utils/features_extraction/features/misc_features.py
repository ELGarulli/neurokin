from typing import List

import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.core_elg import FeatureExtraction


class Height(FeatureExtraction):
    extraction_target = "misc"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: List, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        target_markers_coords = [coord for marker in target_bodyparts for coord in bodyparts_coordinates if
                                 marker in coord]
        for bodypart in target_markers_coords:

            feat = df[bodypart].values
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{bodypart}_height"]))

        df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat
