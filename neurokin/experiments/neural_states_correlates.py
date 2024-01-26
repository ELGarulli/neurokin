import glob
import re
import os
import numpy as np
import pickle as pkl
from typing import List, Dict
from neurokin.utils.helper.load_config import read_config
from neurokin.experiments.neural_correlates import (get_events_dict, get_neural_correlates_dict, get_psd_dict)

"""
class :  Neural correlates of states

inputs
	* (dataset)
	* timeslice
	* stream names: List
	* channel dictionary
	
	* experiment structure

attributes
	+ freqs [autofilled]
	+ psd_dictionary_all_conditions [autofilled]

methods
	# dataset load * file name
	# dataset create * experiment path
	# create dict psds (now pre_process_data)
	# split dataset (authomatic detection of conditions? called already by create dict psds?)
	# sort data animal event (already called in pre_process_data?)
	# average by animal ### what type should it be at this point? dict or df?###



aux conditions -> split in methods or utils function
"""


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
        self.fs: float
        self.events_dataset_dict: Dict
        self.raw_neural_correlates_dict: Dict
        self.psds_correlates_dict: Dict

    def load_dataset(self, dataset_path):
        """
        Load existing dataset dictionary, avoiding waiting time for processing
        :param dataset_path: path including filename to the dataset. Expects pickle files
        :return:
        """
        return pkl.load(open(dataset_path, "rb"))

    def _save_dataset(self, dataset, filename):
        if dataset is None:
            raise ValueError("The dataset has not been initialized")
        with open(f"{filename}.pkl", "wb") as handle:
            pkl.dump(dataset, handle)

    def save_dataset(self, dataset, filename):
        """
        Saves dataset dictionary to pickle file
        :param dataset: str name of which dataset to save
        :param filename: filename without extension
        :return:
        """
        accepted_datasets = ["events_dataset", "raw_neural_correlate_dataset", "psd_neural_correlate_dataset"]
        if dataset not in accepted_datasets:
            raise ValueError(f"Dataset not found. Please select one of the following : {accepted_datasets}")

        if dataset == "events_dataset":
            self._save_dataset(self.events_dataset_dict, filename)
        if dataset == "raw_neural_correlates_dataset":
            self._save_dataset(self.raw_neural_correlates_dict, filename)
        if dataset == "psd_neural_correlates_dataset":
            self._save_dataset(self.psds_correlates_dict, filename)

    def create_events_dataset(self, experiment_path, conditions):
        _conditions = conditions
        all_events_dict = {_condition: {} for _condition in conditions}
        dates = [str(date) for date in self.experiment_structure.keys()]
        for date in dates:
            all_events_dict.setdefault(date, {})
            animals = [animal for animal in self.experiment_structure[int(date)].keys()]
            for animal in animals:
                all_events_dict[date].setdefault(animal, {})
                conditions = [condition for condition in self.experiment_structure[int(date)][animal].keys()]
                for condition in conditions:
                    runs = [str(run) for run in self.experiment_structure[int(date)][animal][condition]]
                    runs = ["0" + run if len(run) == 1 else run for run in runs]
                    for run in runs:
                        run_path = "/".join([experiment_path, date, animal, run]) + "/"

                        try:
                            event_path = [run_path + fname for fname in os.listdir(run_path) if
                                          re.match(r"[a-z_-]+[0-9]{2}.csv", fname)][0]
                            events_dict = get_events_dict(event_path=event_path,
                                                          skiprows=self.skiprows,
                                                          framerate=self.framerate)
                            all_events_dict[date][animal][run] = events_dict
                        except FileNotFoundError as error:
                            print(f"No .csv events File Found For {date}, {animal}, {run}")

        self.events_dataset_dict = all_events_dict

    def create_raw_neural_dataset(self, experiment_path):
        if self.events_dataset_dict is None:
            print("Please create or load an events dictionary first, "
                  "using either the method create_events_dataset or load_dataset"
                  "If the dataset is loaded it should be assigned to the attribute events_dataset_dict")

            return
        dataset_raw_correlates = {}
        dates = [str(date) for date in self.experiment_structure.keys()]

        for date in dates:
            dataset_raw_correlates.setdefault(date, {})
            animals = [animal for animal in self.experiment_structure[int(date)].keys()]
            for animal in animals:
                dataset_raw_correlates[date].setdefault(animal, {})
                conditions = [condition for condition in self.experiment_structure[int(date)][animal].keys()]
                for condition in conditions:
                    runs = [str(run) for run in self.experiment_structure[int(date)][animal][condition]]
                    runs = ["0" + run if len(run) == 1 else run for run in runs]
                    for run in runs:
                        channel_of_interest = self.ch_of_interest[animal]
                        run_path = "/".join([experiment_path, date, animal, run]) + "/"
                        event_dict = self.events_dataset_dict[date][animal][run]

                        raw_neural_correlate, fs = get_neural_correlates_dict(neural_path=run_path,
                                                                              channel_of_interest=channel_of_interest,
                                                                              stream_names=self.stream_names,
                                                                              events_dict=event_dict,
                                                                              time_cutoff=self.timeslice)
                        if fs is not None:
                            self.fs = fs

                        dataset_raw_correlates[date][animal][run] = raw_neural_correlate

        self.raw_neural_correlates_dict = dataset_raw_correlates

    def create_psd_dataset(self, nfft, nov):
        if self.raw_neural_correlates_dict is None:
            print("Please create or load a raw neural dictionary first, "
                  "using either the method create_events_dataset or load_dataset."
                  "If the dataset is loaded it should be assigned to the attribute raw_neural_correlates_dict")
            return

        dataset_psd = {}
        dates = [str(date) for date in self.experiment_structure.keys()]
        self.freqs = None

        for date in dates:
            dataset_psd.setdefault(date, {})
            animals = [animal for animal in self.experiment_structure[int(date)].keys()]
            for animal in animals:
                dataset_psd[date].setdefault(animal, {})
                conditions = [condition for condition in self.experiment_structure[int(date)][animal].keys()]
                for condition in conditions:
                    runs = [str(run) for run in self.experiment_structure[int(date)][animal][condition]]
                    runs = ["0" + run if len(run) == 1 else run for run in runs]
                    for run in runs:
                        neural_dict = self.raw_neural_correlates_dict[date][animal][run]
                        dataset_psd[date][animal][run], freqs_ = get_psd_dict(neural_dict=neural_dict,
                                                                              fs=self.fs,
                                                                              nfft=nfft,
                                                                              noverlap=nov,
                                                                              zscore=True)
                        if freqs_ is not None:
                            self.freqs = freqs_
        self.psds_correlates_dict = dataset_psd

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