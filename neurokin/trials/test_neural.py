import os
from neurokin.neural_data import NeuralData

#
#root = "../temp_data/optogen_1000_left_2022-07-20_15-58-09/"
#node = "/Record Node 104/"
#dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
#nperseg=2**10
#recordings=[]
#for i in dirlist:
#    recording = NeuralData(path=root+i+node)
#    recording.load_open_ephys(experiment="experiment1", recording="recording1", sync_present=True, sync_ch=39)

path = "../../../temp_data/221216_cl_54/"
neural = NeuralData(path=path+"ENWE_00054-221216-085714")
neural.load_tdt_data(stream_name="NPr1", sync_present=True, stim_stream_name="Actm")