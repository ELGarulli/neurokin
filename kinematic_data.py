from numpy.typing import ArrayLike
from utils.kinematics import c3d_import_export, event_detection, kinematics_processing
import pandas as pd


class KinematicDataRun:
    """
    This class represents the kinematics data recorded in a single run.
    """

    def __init__(self, path):
        self.path = path
        self.gait_cycles_start: ArrayLike
        self.gait_cycles_end: ArrayLike
        self.fs: float
        self.condition: str
        self.markers_df: pd.DataFrame
        self.left_mtp_lift: ArrayLike = None
        self.left_mtp_land: ArrayLike = None
        self.left_mtp_max: ArrayLike = None
        self.right_mtp_lift: ArrayLike = None
        self.right_mtp_land: ArrayLike = None
        self.right_mtp_max: ArrayLike = None

    def load_kinematics(self,
                        correct_shift: bool = False,
                        correct_tilt: bool = False,
                        to_shift: ArrayLike = None,
                        to_tilt: ArrayLike = None,
                        shift_reference_marker: str = "",
                        tilt_reference_marker: str = ""):
        self.markers_df = c3d_import_export.import_c3d(self.path)
        if correct_shift:
            # TODO add try except clause if ref marker doesnt exist or col name in to shift
            self.markers_df = kinematics_processing.shift_correct(self.markers_df, shift_reference_marker, to_shift)
        if correct_tilt:
            # TODO add try except clause if ref marker doesnt exist or col name in to tilt
            self.markers_df = kinematics_processing.tilt_correct(self.markers_df, tilt_reference_marker, to_tilt)
        return

    def compute_gait_cycles_timestamp(self, left_marker, right_marker):
        self.left_mtp_lift, self.left_mtp_land, self.left_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[left_marker])
        self.right_mtp_lift, self.right_mtp_land, self.right_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[right_marker])

        return
