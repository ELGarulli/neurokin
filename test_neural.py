from neural_data import NeuralData
from matplotlib import pyplot as plt

neural_data = NeuralData("../temp_data/220816_discorat/ENWE_00052-220816-132934")
neural_data.load_tdt_data(sync_ch=True, stream_name="Wav2", stim_stream_name="Wav1")

plt.plot(neural_data.sync_data)
plt.show()
print("")