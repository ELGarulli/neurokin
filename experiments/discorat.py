from neural_data import NeuralData
from utils.neural.processing import get_stim_timestamps
class Discorat:
    def __init__(self, neural_data):
        self.synch = neural_data.synch_data
        self.raw = neural_data.raw



    def parse_around_flash(self):
        get_stim_timestamps(self.synch, )
        return