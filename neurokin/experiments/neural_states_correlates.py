import os
import re
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurokin.experiments.neural_correlates import (get_events_dict, get_neural_correlates_dict, get_psd_dict,
                                                    get_psd_single_event_type)
from neurokin.utils.experiments.neural_states_helper import (get_events_per_animal, condense_neural_event_types,
                                                             get_per_animal_psds_df, drop_subject_id, save_data,
                                                             compute_events_percentage,
                                                             condense_distribution_event_types,
                                                             get_group_split, get_state_graph_stats,
                                                             get_runs_list)
from neurokin.utils.helper.load_config import read_config
from neurokin.utils.experiments.neural_correlates_plot import plot_psd_single_state


class NeuralCorrelatesStates():

    def __init__(self,
                 timeslice: float,
                 experiment_structure_filepath: str,
                 skip_subjects: List[str] = [],
                 skip_conditions: List[str] = [],
                 skiprows: int = 2,
                 framerate: float = 200):

        self.timeslice = timeslice
        self.experiment_structure = read_config(experiment_structure_filepath, converts_keys_to_string=True)
        self.skip_subjects = skip_subjects
        self.skip_conditions = skip_conditions
        self.skiprows = skiprows
        self.framerate = framerate
        self.stream_names: List[str] = []
        self.ch_of_interest: Dict[str, int] = {}
        self.freqs: np.array = None
        self.fs: float = None
        self.events_dataset: pd.DataFrame = None
        self.raw_neural_correlates_dataset: pd.DataFrame = None
        self.psds_correlates_dataset: pd.DataFrame = None

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
            save_data(self.events_dataset, filename)
        if dataset == "raw_neural_correlates_dataset":
            save_data(self.fs, "fs")
            save_data(self.raw_neural_correlates_dataset, filename)
        if dataset == "psd_neural_correlates_dataset":
            save_data(self.psds_correlates_dict, filename)
            save_data(self.freqs, "freqs")

    def create_events_dataset(self, experiment_path, verbose=False):
        """
        Takes the experiment structure and based on the runs listed there it looks for .csv files to import.
        Then based on the labels, creates a dataframe containing condition, date, animal, run and event.
        Each event contains the list of timestamps [start, end] for that specific run
        :param experiment_path: folder where to find the dataset
        :param verbose: if True it will print the currently processed run
        :return: dataframe containing data of each trial  and timestamps of events
        :param experiment_path:
        """
        trial_list = get_runs_list(self.experiment_structure, self.skip_subjects, self.skip_conditions)
        trial_data = []
        for trial in trial_list:
            date, subject, condition, run = trial
            run_path = "/".join([experiment_path, date, subject, run]) + "/"

            if verbose:
                print(f"Currently processing: {date} - {subject} - {condition} - {run}")
            try:
                event_path = [run_path + fname for fname in os.listdir(run_path) if
                              re.match(r"(?i)[a-z_-]+[0-9]{1,3}.csv", fname)][0]
                events_dict = get_events_dict(event_path=event_path,
                                              skiprows=self.skiprows,
                                              framerate=self.framerate)
                events = [events_dict["gait"],
                          events_dict["nlm_rest"],
                          events_dict["nlm_active"],
                          events_dict["fog_rest"],
                          events_dict["fog_active"]]

                trial_data.append(trial + events)

            except FileNotFoundError as error:
                print(f"{error} No .csv events File Found For {date}, {subject}, {run}")

        df = pd.DataFrame(trial_data, columns=["date",
                                               "subject",
                                               "condition",
                                               "run",
                                               "event_gait",
                                               "event_nlm_rest",
                                               "event_nlm_active",
                                               "event_fog_rest",
                                               "event_fog_active"])
        self.events_dataset = df
        return

    def create_raw_neural_dataset(self, experiment_path, stream_names: List[str], ch_of_interest: Dict[str, int],
                                  verbose=False):

        if self.events_dataset is None:
            print("Please create or load an events dictionary first, "
                  "using either the method create_events_dataset or load_dataset"
                  "If the dataset is loaded it should be assigned to the attribute events_dataset_dict")

            return
        self.stream_names = stream_names
        self.ch_of_interest = ch_of_interest
        trial_list = get_runs_list(self.experiment_structure, self.skip_subjects, self.skip_conditions)
        trial_neural_data = []
        for trial in trial_list:
            date, subject, condition, run = trial
            run_path = "/".join([experiment_path, date, subject, run]) + "/"
            channel_of_interest = self.ch_of_interest[subject]
            neural_event_df = self.events_dataset[(self.events_dataset["date"] == date) &
                                                  (self.events_dataset["subject"] == subject) &
                                                  (self.events_dataset["condition"] == condition) &
                                                  (self.events_dataset["run"] == run)]
            if verbose:
                print(f"Currently processing: {date} - {subject} - {condition} - {run}")

            raw_neural_correlate, fs = get_neural_correlates_dict(neural_path=run_path,
                                                                  channel_of_interest=channel_of_interest,
                                                                  stream_names=stream_names,
                                                                  events_df=neural_event_df,
                                                                  time_cutoff=self.timeslice)
            if fs is not None:
                self.fs = fs

            events = [raw_neural_correlate["event_gait"],
                      raw_neural_correlate["event_nlm_rest"],
                      raw_neural_correlate["event_nlm_active"],
                      raw_neural_correlate["event_fog_rest"],
                      raw_neural_correlate["event_fog_active"]]

            trial_neural_data.append(trial + events)

        df = pd.DataFrame(trial_neural_data, columns=self.events_dataset.columns)
        self.raw_neural_correlates_dataset = df

    def create_psd_dataset(self, nfft, nov, zscore=False):
        """
        Computes the Power Spectra Density of the neural correlates.
        :param nfft: NFFT parameter to use for the Fourier Transform
        :param nov: Overlap parameter to use for the Fourier Transform
        :param verbose: if True it will print the currently processed run
        :return: dictionary structured as condition, date, animal, run, event containing the PSD neural correlates.
        """
        if self.raw_neural_correlates_dataset is None:
            print("Please create or load a raw neural dictionary first, "
                  "using either the method create_events_dataset or load_dataset."
                  "If the dataset is loaded it should be assigned to the attribute raw_neural_correlates_dict")
            return
        events_columns = [c for c in self.raw_neural_correlates_dataset.columns if c.startswith("event")]
        meta_columns = [c for c in self.raw_neural_correlates_dataset.columns if not c.startswith("event")]
        psds_correlates_dataset = self.raw_neural_correlates_dataset[events_columns].applymap(
            get_psd_single_event_type,
            fs=self.fs,
            nfft=nfft,
            noverlap=nov,
            zscore=zscore)
        self.psds_correlates_dataset = pd.concat((self.raw_neural_correlates_dataset[meta_columns],
                                                  psds_correlates_dataset), axis=1)
        self.freqs = np.fft.rfftfreq(n=nfft, d=1 / self.fs)

    # PANDIZE
    def _create_psd_dataset(self, nfft, nov, verbose=False):
        """
        Computes the Power Spectra Density of the neural correlates.
        :param nfft: NFFT parameter to use for the Fourier Transform
        :param nov: Overlap parameter to use for the Fourier Transform
        :param verbose: if True it will print the currently processed run
        :return: dictionary structured as condition, date, animal, run, event containing the PSD neural correlates.
        """
        if self.raw_neural_correlates_dataset is None:
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
                        if verbose:
                            print(f"Currently processing: {date} - {animal} - {condition} - {run}")
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
        """

        :param test_sbj_list:
        :param condense:
        :return:
        """
        per_animal_avg = get_per_animal_psds_df(self.psds_correlates_dataset, condense=condense)
        group_split = get_group_split(test_sbj_list=test_sbj_list, df=per_animal_avg)
        return group_split

    # PANDIZE
    def _plot_prep_psds_dataset(self, test_sbj_list, condense=True):
        """
        Fixed shortcut to generate a PSDs dictionary dataset ready to be plotted
        :param test_sbj_list: list of subject IDs that belong to the test group
        :param condense: bool if to condense from 5 categories to 3
        :return: PSDs dictionary dataset structured as group, condition, state
        """
        per_animal_events = get_events_per_animal(self.psds_correlates_dict)
        if condense:
            per_animal_events = condense_neural_event_types(per_animal_events)
        per_animal_avg = get_per_animal_psds_df(per_animal_events)
        no_sbj_id = drop_subject_id(test_sbj_list=test_sbj_list, animals_avg_psds=per_animal_avg)
        return no_sbj_id

    def plot_prep_states_distribution(self, test_sbj_list, condense=True, stat="std"):
        """
        Fixed shortcut to generate a stats dictionary of the state distribution, ready to be plotted
        :param test_sbj_list: list of subject IDs that belong to the test group
        :param condense:  bool if to condense from 5 categories to 3
        :return: stats dictionary dataset structured as group, condition,
                state, stats {"mean":, "upper_bound", "lower_bound}
        """
        events_percentage = compute_events_percentage(self.events_dataset)
        if condense:
            events_percentage = condense_distribution_event_types(events_percentage)
        group_split = get_group_split(test_sbj_list=test_sbj_list, df=events_percentage)
        stats = get_state_graph_stats(group_cond_df=group_split, stat=stat)
        return stats


if __name__ == "__main__":
    NFFT = 2 ** 12
    NOV = int(NFFT / 4)
    TIME_CUTOFF = 1.5
    experiment_structure_path = "../../../analysis/neural_correlates_states_clean/experiment_structure.yaml"
    pda = ["NWE00053", "NWE00054", "NWE00130", "NWE00160", "NWE00161", "NWE00162", "NWE00163", "NWE00164"]
    skip_animals = ["NWE00053", "NWE00054", "NWE00052"]

    ncs = NeuralCorrelatesStates(timeslice=TIME_CUTOFF,
                                 experiment_structure_filepath=experiment_structure_path,
                                 skip_subjects=skip_animals)

    ncs.fs = 24414.1
    ncs.raw_neural_correlates_dataset = pd.read_pickle(
        "../../../analysis/neural_correlates_states_clean/raw_neural.pkl")
    ncs.create_psd_dataset(NFFT, NOV, zscore=False)
    df = ncs.plot_prep_psds_dataset(test_sbj_list=pda, condense=True)

    fig, ax = plt.subplots()
    plot_psd_single_state(ax, df, group=True,
                                                 condition="baseline",
                                                 state="event_gait",
                                                 freqs=ncs.freqs,
                                                 color="crimson",
                                                 idx_min=2,
                                                 idx_max=75)

    print("")
