from typing import Dict

import numpy as np
import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.core import FeatureExtraction


class Angle(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = self.angle(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle"]))

        df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat

    def angle(self, vectors):
        if vectors.shape[1] == 9:
            a, b, c = vectors[:, :3], vectors[:, 3:6], vectors[:, 6:9]
        elif vectors.shape[1] == 6:
            a, b, c = vectors[:, 2], vectors[:, 2:4], vectors[:, 4:6]

        bas = a - b
        bcs = c - b
        angles = []
        for ba, bc in zip(bas, bcs):
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angles.append(angle)
        return np.array(angles)
