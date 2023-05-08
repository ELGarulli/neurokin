from typing import Tuple, List, Dict, Optional, Union, Any
import pandas as pd
from .core import FeatureExtraction, DefaultParams
from neurokin.utils.kinematics.gait_params_basics import get_phase, get_angle
from neurokin.utils.features.joint_angles_dlc2kin import JointAnglesDLC


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
            self,
            source_marker_ids: List[str],
            marker_df: pd.DataFrame,
            params: Dict[str, Any],
    ) -> pd.DataFrame:

    # filter df for specific columns, raise error if angles not calculated yet
    joint = list(source_marker_ids.keys())[0]
    names = marker_df.columns.names
    scorer = marker_df.columns.get_level_values("scorer")[0]
    angels_df = marker_df[scorer, joint, "angle"]
    angels_df = angels_df.to_frame(name="angular_velocity")
    angels_df.columns = pd.MultiIndex.from_product(
        [[scorer], source_marker_ids.keys(), angels_df.columns], names=names)

    df_angular_momentum = dlc2kinematics.compute_joint_velocity(
        joint_angle=angels_df,
        filter_window=params["window_size"], save=False)
    self._assert_valid_output(output_df=df_angular_momentum, marker_df=marker_df)
    return df_angular_momentum