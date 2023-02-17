from typing import Dict
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
import numpy as np
import matlab.engine
from neurokin.utils.kinematics import ggait_compliance
from neurokin.constants.matlab_engine import GENPATH
import logging

logger = logging.getLogger(__name__)


class GGait:
    """
    This class interfaces with the GGait MATLAB code, use at your own risk.
    """

    def __init__(self, filename, path, fs):

        self.path = path

        self.gait_cycles_start: ArrayLike
        self.gait_cycles_end: ArrayLike
        self.fs = fs
        self.h: Dict
        self.filename = filename
        self.eng = None

        self.gait_cycle_dict_left: None
        self.gait_cycle_dict_left: None

    def load_kinematics(self):
        try:
            names = matlab.engine.find_matlab()
            self.eng = matlab.engine.start_matlab(names[0])

        except IndexError as error:
            logger.error(error)
            print("WARNING: Make sure MATLAB engine is connected!")
            raise

        paths_to_all_ggait_folders = self.eng.genpath(GENPATH)
        self.eng.addpath(paths_to_all_ggait_folders, nargout=0)
        self.h = ggait_compliance.load_raw_kinematics(self.eng, self.filename, self.path)
        return

    def compute_gait_cycles_timestamp(self):
        self.gait_cycle_dict_left = ggait_compliance.get_gait_cycle_bounds(h=self.h, data_name="Data_L", fs=self.fs)
        self.gait_cycle_dict_right = ggait_compliance.get_gait_cycle_bounds(h=self.h, data_name="Data_R", fs=self.fs)

    def get_gait_param(self):

        self.h = self.eng.minEx_2(self.h,
                                  self.h['Data_L'],
                                  self.h['Data_R'],
                                  self.h['TIME'],
                                  matlab.int32(self.gait_cycle_dict_left["toe_off"]),
                                  matlab.int32(self.gait_cycle_dict_right["toe_off"]),
                                  matlab.int32(self.gait_cycle_dict_left["heel_strike"]),
                                  matlab.int32(self.gait_cycle_dict_right["heel_strike"]),
                                  0.0,
                                  0.0,
                                  nargout=1)

        return

    def print_step_partition(self, output_folder, run):
        """
        Prints the gait cycles bounds to a png image
        :param output_folder: where to store the images
        :param run: run name to include in the name
        :return:
        """
        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=2.0)
        filename = output_folder + run + "_steps_partition.png"

        raw_data = self.h["Data_L"]
        raw_data = np.array(raw_data)
        axs[0].plot(raw_data)
        min_, max_ = axs[0].get_ylim()
        axs[0].vlines(self.gait_cycle_dict_left["toe_off"], min_, max_, colors="green")
        axs[0].vlines(self.gait_cycle_dict_left["heel_strike"], min_, max_, colors="red")
        axs[0].set_title("Left side")

        raw_data = self.h["Data_R"]
        raw_data = np.array(raw_data)
        axs[1].plot(raw_data)
        min_, max_ = axs[1].get_ylim()
        axs[1].vlines(self.gait_cycle_dict_right["toe_off"], min_, max_, colors="green")
        axs[1].vlines(self.gait_cycle_dict_right["heel_strike"], min_, max_, colors="red")
        axs[1].set_title("Right side")
        plt.savefig(filename, facecolor="white")
        plt.close()

    def test_minEx3(self):
        self.h = self.eng.minEx_3(self.h)
        return
