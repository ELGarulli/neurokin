import re
import os
import numpy as np
from typing import List, Dict
from neurokin.utils.helper.load_config import read_config
from neurokin.experiments.neural_correlates import (get_events_dict, get_neural_correlates_dict, get_psd_dict)
from neurokin.utils.experiments.neural_states_helper import (get_events_per_animal, condense_neural_event_types,
                                                             get_per_animal_average, drop_subject_id, save_data,
                                                             compute_events_percentage,
                                                             condense_distribution_event_types,
                                                             get_group_split, compute_state_distribution_stats)


class NeuralCorrelatesStates():

    def __init__(self,
                 timeslice: float,
                 experiment_structure_filepath: str,
                 skip_subjects: List[str] = [],
                 skiprows: int = 2,
                 framerate: float = 200):

        self.timeslice = timeslice

        self.experiment_structure = read_config(experiment_structure_filepath)
        self.skip_subjects = skip_subjects
        self.skiprows = skiprows
        self.framerate = framerate
        self.stream_names: List[str]
        self.ch_of_interest: Dict[str, int]
        self.freqs: np.array
        self.fs: float
        self.events_dataset_dict: Dict
        self.raw_neural_correlates_dict: Dict
        self.psds_correlates_dict: Dict

    def save_dataset(self, dataset, filename):
        """
        Saves dataset dictionary to pickle file
        :param dataset: str name of which dataset to save
        :param filename: filename without extension
        :return:
        """
        accepted_datasets = ["events_dataset", "raw_neural_correlates_dataset", "psd_neural_correlates_dataset"]
        if dataset not in accepted_datasets:
            raise ValueError(f"Dataset not found. Please select one of the following : {accepted_datasets}")

        if dataset == "events_dataset":
            save_data(self.events_dataset_dict, filename)
        if dataset == "raw_neural_correlates_dataset":
            save_data(self.fs, "fs")
            save_data(self.raw_neural_correlates_dict, filename)
        if dataset == "psd_neural_correlates_dataset":
            save_data(self.psds_correlates_dict, filename)
            save_data(self.freqs, "freqs")

    def create_events_dataset(self, experiment_path, conditions_of_interest):
        all_events_dict = {c: {} for c in conditions_of_interest}
        dates = [str(date) for date in self.experiment_structure.keys()]
        for date in dates:
            animals = [animal for animal in self.experiment_structure[int(date)].keys()]
            animals = [animal for animal in animals if animal not in self.skip_subjects]
            for animal in animals:
                conditions = [condition for condition in self.experiment_structure[int(date)][animal].keys()]
                for condition in conditions:
                    if condition in conditions_of_interest:
                        all_events_dict[condition].setdefault(date, {})
                        all_events_dict[condition][date].setdefault(animal, {})
                        runs = [str(run) for run in self.experiment_structure[int(date)][animal][condition]]
                        runs = ["0" + run if len(run) == 1 else run for run in runs]
                        for run in runs:
                            run_path = "/".join([experiment_path, date, animal, run]) + "/"

                            try:
                                event_path = [run_path + fname for fname in os.listdir(run_path) if
                                              re.match(r"(?i)[a-z_-]+[0-9]{2}.csv", fname)][0]
                                events_dict = get_events_dict(event_path=event_path,
                                                              skiprows=self.skiprows,
                                                              framerate=self.framerate)
                                all_events_dict[condition][date][animal][run] = events_dict
                            except FileNotFoundError as error:
                                print(f"No .csv events File Found For {date}, {animal}, {run}")

        self.events_dataset_dict = all_events_dict

    def create_raw_neural_dataset(self, experiment_path, stream_names: List[str], ch_of_interest: Dict[str, int]):
        if self.events_dataset_dict is None:
            print("Please create or load an events dictionary first, "
                  "using either the method create_events_dataset or load_dataset"
                  "If the dataset is loaded it should be assigned to the attribute events_dataset_dict")

            return
        self.stream_names = stream_names
        self.ch_of_interest = ch_of_interest
        conditions_of_interest = list(self.events_dataset_dict.keys())
        dataset_raw_correlates = {c: {} for c in conditions_of_interest}

        for condition in conditions_of_interest:
            dates = list(self.events_dataset_dict[condition].keys())
            for date in dates:
                animals = list(self.events_dataset_dict[condition][date].keys())
                for animal in animals:
                    runs = list(self.events_dataset_dict[condition][date][animal])
                    for run in runs:
                        dataset_raw_correlates[condition].setdefault(date, {})
                        dataset_raw_correlates[condition][date].setdefault(animal, {})
                        channel_of_interest = self.ch_of_interest[animal]
                        run_path = "/".join([experiment_path, date, animal, run]) + "/"
                        event_dict = self.events_dataset_dict[condition][date][animal][run]

                        raw_neural_correlate, fs = get_neural_correlates_dict(neural_path=run_path,
                                                                              channel_of_interest=channel_of_interest,
                                                                              stream_names=stream_names,
                                                                              events_dict=event_dict,
                                                                              time_cutoff=self.timeslice)
                        if fs is not None:
                            self.fs = fs

                        dataset_raw_correlates[condition][date][animal][run] = raw_neural_correlate

        self.raw_neural_correlates_dict = dataset_raw_correlates

    def create_psd_dataset(self, nfft, nov):
        if self.raw_neural_correlates_dict is None:
            print("Please create or load a raw neural dictionary first, "
                  "using either the method create_events_dataset or load_dataset."
                  "If the dataset is loaded it should be assigned to the attribute raw_neural_correlates_dict")
            return

        conditions_of_interest = list(self.raw_neural_correlates_dict.keys())
        dataset_psd = {c: {} for c in conditions_of_interest}

        self.freqs = None

        for condition in conditions_of_interest:
            dates = list(self.raw_neural_correlates_dict[condition].keys())
            for date in dates:
                animals = list(self.raw_neural_correlates_dict[condition][date].keys())
                for animal in animals:
                    runs = list(self.raw_neural_correlates_dict[condition][date][animal])
                    for run in runs:
                        dataset_psd[condition].setdefault(date, {})
                        dataset_psd[condition][date].setdefault(animal, {})
                        neural_dict = self.raw_neural_correlates_dict[condition][date][animal][run]
                        dataset_psd[condition][date][animal][run], freqs_ = get_psd_dict(neural_dict=neural_dict,
                                                                                         fs=self.fs,
                                                                                         nfft=nfft,
                                                                                         noverlap=nov,
                                                                                         zscore=True)
                        if freqs_ is not None:
                            self.freqs = freqs_
        self.psds_correlates_dict = dataset_psd

    def plot_prep_psds_dataset(self, test_sbj_list, condense=True):
        per_animal_events = get_events_per_animal(self.psds_correlates_dict)
        if condense:
            per_animal_events = condense_neural_event_types(per_animal_events)
        per_animal_avg = get_per_animal_average(per_animal_events)
        no_sbj_id = drop_subject_id(test_sbj_list=test_sbj_list, animals_avg_psds=per_animal_avg)
        return no_sbj_id

    def plot_prep_states_distribution(self, test_sbj_list, condense=True):
        events_percentage = compute_events_percentage(self.events_dataset_dict)
        if condense:
            events_percentage = condense_distribution_event_types(events_percentage)
        group_split = get_group_split(test_sbj_list=test_sbj_list, animals_avg_dataset=events_percentage)
        stats = compute_state_distribution_stats(group_split)
        return stats
