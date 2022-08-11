from typing import Dict
from numpy.typing import ArrayLike
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


    def load_kinematics(self):
        try:
            names = matlab.engine.find_matlab()
            eng = matlab.engine.start_matlab(names[0])

        except:
            print("Make sure MATLAB engine is connected")
            return None
        paths_to_all_ggait_folders = eng.genpath("C:/Users/Elisa/Documents/GitHub/gait-new-repo/")
        eng.addpath(paths_to_all_ggait_folders, nargout=0)
        self.h = gait.load_raw_kinematics(eng, self.filename, self.path)
        return

    def compute_gait_cycles_timestamp(self):
        self.gait_cycle_dict_left = gait.get_gait_cycle_bounds(h=self.h, data_name="Data_L")
        self.gait_cycle_dict_right = gait.get_gait_cycle_bounds(h=self.h, data_name="Data_R")
        print(self.gait_cycle_dict_left)


    def get_gait_param(self):
        # h = eng.minEx_2(h, h['Data_L'], h['Data_R'],
        #                h['TIME'],
        #                steps_dict['left_toe_off'],
        #                steps_dict['right_toe_off'],
        #                steps_dict['left_heel_strike'],
        #                steps_dict['right_heel_strike'],
        #                steps_dict['uneven_l'],
        #                steps_dict['uneven_r'], nargout=1)
    return
