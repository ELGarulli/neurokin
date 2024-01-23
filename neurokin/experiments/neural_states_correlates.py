import glob
import numpy as np
import pickle as pkl
from typing import List, Dict
from neurokin.utils.helper.load_config import read_config
from neurokin.experiments.neural_correlates import (get_events_dict, get_neural_correlates_dict)

class NeuralCorrelatesStates():

    def __init__(self, timeslice: float,
                 stream_names: List[str],
                 ch_of_interest: Dict[str, int],
                 experiment_structure_filepath: str,
                 skiprows: int = 2,
                 framerate: float = 200):

        self.timeslice = timeslice
        self.stream_names = stream_names
        self.ch_of_interest = ch_of_interest
        self.experiment_structure = read_config(experiment_structure_filepath)
        self.skiprows = skiprows
        self.framerate = framerate
        self.freqs: np.array
        self.dataset_dict: Dict

    def load_dataset(self, dataset_path):
        """
        Load existing dataset dictionary and sets the attribute dataset_dict to it, avoiding waiting time for processing
        :param dataset_path: path including filename to the dataset. Expects pickle files
        :return:
        """
        self.dataset_dict = pkl.load(open(dataset_path, "rb"))

    def import_dataset(self, experiment_path):
        dataset_psd = {}
        dates = [str(date) for date in self.experiment_structure.keys()]
        freqs = None

        for date in dates:
            dataset_psd.setdefault(date, {})
            animals = [animal for animal in self.experiment_structure[date].keys()]
            for animal in animals:
                dataset_psd[date].setdefault(animal, {})
                runs = [str(run) for run in self.experiment_structure[date][animal].keys()]
                runs = ["0" + run if len(run) == 1 else run for run in runs]
                for run in runs:
                    channel_of_interest = self.ch_of_interest[animal]
                    run_path = "/".join([experiment_path, date, animal, run]) + "/"
                    event_path = glob.glob(run_path + "*.csv")[0]

                    events_dict = get_events_dict(event_path=event_path,
                                                  skiprows=self.skiprows,
                                                  framerate=self.framerate)
                    neural_dict, fs = get_neural_correlates_dict(neural_path=run_path)

        return

    def process_dataset(self):
        return

    def __create_dict_psds(self):
        return

    def __split_dataset_by_condition(self):
        return

    def __sort_data_animal_event(self):
        return

    def __average_by_animal(self):
        return
