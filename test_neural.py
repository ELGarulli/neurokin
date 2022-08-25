import numpy as np

from neural_data import NeuralData
from matplotlib import pyplot as plt
from utils.exporting import export_neural_data_to_bin
neural_data = NeuralData("../temp_data/220816_discorat/ENWE_00052-220816-132934")
neural_data.load_tdt_data(sync_ch=False, stream_name="Wav2", stim_stream_name="Wav1")

#export_neural_data_to_bin(neural_data.raw, "test.dat")
n_samples = len(neural_data.raw[0])
