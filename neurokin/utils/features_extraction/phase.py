from typing import Tuple, List, Dict, Optional, Union, Any
import pandas as pd
from .core import FeatureExtraction, DefaultParams
from neurokin.utils.kinematics.gait_params_basics import get_phase, get_angle
from neurokin.utils.features_extraction.joint_angles_dlc2kin import JointAnglesDLC


class PhasesAngle(FeatureExtraction):
    # input_type = "joints"

    @property
    def input_type(self) -> str:
        return "joints"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {}
        return default_types

    def _run_feature_extraction(
            self, source_marker_ids: List[str], marker_df, params
    ) -> pd.DataFrame:
        markers_and_features_df = marker_df.copy()
        joint = [key for key in source_marker_ids.keys()][0]
        if not joint in marker_df.columns.levels[1]:
            extractor_obj = JointAnglesDLC()
            feature = extractor_obj.extract_features(
                source_marker_ids=source_marker_ids, marker_df=marker_df, params=params
            )
            markers_and_features_df = pd.concat(
                (markers_and_features_df, feature), axis=1
            )

        angle = markers_and_features_df["scorer"][joint]
        phase_df = angle.apply(get_phase)

        filtered_df = self._rename_columns_on_selected_idx_level(
            df=phase_df, suffix="_phase"
        )
        return filtered_df
