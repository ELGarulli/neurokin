from typing import Tuple, List, Dict, Optional, Union, Any
import pandas as pd
from .core import FeatureExtraction, DefaultParams
from neurokin.utils.kinematics.gait_params_basics import get_phase, get_angle
from dlc2kinematics.joint_analysis import compute_joint_angles

class PhasesAngle(FeatureExtraction):
    #input_type = "joints"

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

    def _run_feature_extraction(self, source_marker_ids: List[str]) -> pd.DataFrame:
        #filtered_df = self._copy_filtered_columns_of_df(df_to_filter=self.marker_df,
        #                                                marker_id_filter=source_marker_ids)
        #filtered_df = self.marker_df.droplevel(0, axis=1)
        source_marker_ids = source_marker_ids
        #filtered_df = filtered_df[source_marker_ids]
        source_marker_ids = {"lknee": ["", "", ""]}

        #if not lknee_angle" in self.marker_df.columns():
            #call to angle class
            # append to self.marker_df

        #get phase


        angle = compute_joint_angles(self.marker_df, source_marker_ids,
                                     save=False).to_numpy().flatten()
        filtered_df = get_phase(angle)

        filtered_df = self._rename_columns_on_selected_idx_level(df=filtered_df,
                                                                 suffix='_phase_angle_')
        return filtered_df
