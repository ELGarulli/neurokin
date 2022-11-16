import numpy as np
import os
from neural_data import NeuralData
from matplotlib import pyplot as plt
from utils.exporting import export_neural_data_to_bin
#
root = "../temp_data/optogen_1000_left_2022-07-20_15-58-09/"
node = "/Record Node 104/"
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
nperseg=2**10
recordings=[]
for i in dirlist:
    recording = NeuralData(path=root+i+node)
    recording.load_open_ephys(experiment="experiment1", recording="recording1", sync_present=True, sync_ch=39)