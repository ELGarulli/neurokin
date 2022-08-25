from typing import Dict
from numpy.typing import ArrayLike
import numpy as np
import matlab.engine
from utils.kinematics import gait


class KinematicData:
    """
    This class represents the kinematics data recorded in a single experiment.
    """

    def __init__(self, filename, path):

        self.path = path

        self.gait_cycles_start: ArrayLike
        self.gait_cycles_end: ArrayLike
        self.fs: float
        self.h: Dict
        self.filename = filename
        self.eng = None

    def load_kinematics(self):
        #TODO set better except check which code is raised
        try:
            names = matlab.engine.find_matlab()
            self.eng = matlab.engine.start_matlab(names[0])

        except:
            print("Make sure MATLAB engine is connected")
            return None
        paths_to_all_ggait_folders = self.eng.genpath("C:/Users/Elisa/Documents/GitHub/gait-new-repo/")
        self.eng.addpath(paths_to_all_ggait_folders, nargout=0)
        self.h = gait.load_raw_kinematics(self.eng, self.filename, self.path)
        return

    def compute_gait_cycles_timestamp(self):
        self.gait_cycle_dict_left = gait.get_gait_cycle_bounds(h=self.h, data_name="Data_L")
        self.gait_cycle_dict_right = gait.get_gait_cycle_bounds(h=self.h, data_name="Data_R")

    def get_gait_param(self):

        self.h = self.eng.minEx_2(self.h,
                                  self.h['Data_L'],
                                  self.h['Data_R'],
                                  self.h['TIME'],
                                  matlab.int64(self.gait_cycle_dict_left["toe_off"]),
                                  matlab.int64(self.gait_cycle_dict_right["toe_off"]),
                                  matlab.int64(self.gait_cycle_dict_left["heel_strike"]),
                                  matlab.int64(self.gait_cycle_dict_right["heel_strike"]),
                                  0.0,
                                  0.0,
                                  nargout=1)

        return

    def test_minEx3(self):
        self.h = self.eng.minEx_3(self.h)
        return
