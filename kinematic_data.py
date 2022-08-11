from typing import Dict
from numpy.typing import ArrayLike
import matlab.engine
from utils.kinematics import gait


class KinematicData:
    """
    This class represents the kinematics data recorded in a single experiment.
    """

    def __init__(self, path):

        self.path = path

        self.gait_cycles_start: ArrayLike
        self.gait_cycles_end: ArrayLike
        self.fs: float

    def load_kinematics(self):
        try:
            names = matlab.engine.find_matlab()
            eng = matlab.engine.start_matlab()
        except:
            print("Make sure MATLAB engine is connected")
        gait.load_raw_kinematics()
        return

    def compute_gait_cycles_timestamp(self):
        self.gait_cycles_start, self.gait_cycles_end = gait.get_gait_cycle_bounds()

