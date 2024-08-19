import pickle as pkl

import numpy as np
import pandas as pd


def load_dataset(dataset_path):
    """
    Load existing dataset dictionary, avoiding waiting time for processing

    :param dataset_path: path including filename to the dataset. Expects pickle files
    :return:
    """
    return pkl.load(open(dataset_path, "rb"))


def save_data(data, filename):
    """
    Saves data to pickle files, if the data is not empty. Else returns a ValueError.
    :param data:
    :param filename: filename with path.
    :return:
    """
    if data is None:
        raise ValueError("The dataset has not been initialized")
    with open(f"{filename}.pkl", "wb") as handle:
        pkl.dump(data, handle)


def condense_distribution_event_types(percentage_states_df):
    """
    Condenses from the dataset five categories analysis
    ["fog_active", "fog_rest", "nlm_active", "nlm_rest", gait] to only
    three categories ["fog", "nlm", "gait"]

    :param cond_animal_event_dict: states dictionary structured as condition, animal, events.
    :return:  same dictionary with 3 conditions instead of 5
    """
    five2three = percentage_states_df.copy()
    five2three["event_fog"] = five2three["event_fog_active"] + five2three["event_fog_rest"]
    five2three["event_nlm"] = five2three["event_nlm_active"] + five2three["event_nlm_rest"]
    five2three.drop(["event_fog_active", "event_nlm_active", "event_fog_rest", "event_nlm_rest"], inplace=True, axis=1)
    return five2three


def mean_psds(psds_list):
    """
    Returns mean psds if the list is full

    :param psds_list: list of power spectral density arrays
    :return: mean on axis 0
    """
    if psds_list:
        return np.mean(psds_list, axis=0)


def get_per_animal_psds_df(psds_dataset, condense=False):
    """
    Given the overall psds dataset, computes the average values, per state, per animal

    :param psds_dataset: overall dataset
    :param condense: wheter to condense to the 3 states format or keep the 5 format
    :return: grouped-by-animal dataset
    """
    psds_means = psds_dataset.copy()
    events_col = [c for c in psds_means.columns if c.startswith("event")]
    psds_means.drop(["date", "run"], inplace=True, axis=1)
    psds_means = psds_means.groupby(["subject", "condition"], as_index=False)[events_col].sum(min_count=1)
    psds_means = psds_means.applymap(lambda x: [] if x is None else x)

    if condense:
        psds_means["event_nlm"] = psds_means["event_nlm_rest"] + psds_means["event_nlm_active"]
        psds_means["event_fog"] = psds_means["event_fog_rest"] + psds_means["event_fog_active"]
        psds_means.drop(["event_nlm_rest", "event_fog_rest", "event_nlm_active", "event_fog_active"],
                        inplace=True, axis=1)

    events_col = [c for c in psds_means.columns if c.startswith("event")]
    psds_means[events_col] = psds_means[events_col].applymap(mean_psds, na_action="ignore")
    return psds_means


def get_group_split(test_sbj_list, df):
    """
    Splits the dataset in two groups depending on which subjects are in the test group and which not (control)

    :param test_sbj_list: list of subject IDs that belong to teh test group
    :param animals_avg_dataset: dictionary containing the psds organized by condition, animal and event
    :return: split dictionary between test_group and sham_group
    """
    subject_groups = df.copy()
    subject_groups["group"] = subject_groups["subject"].apply(lambda x: x in test_sbj_list)
    return subject_groups


def compute_duration(timestamps_lists):
    time = 0
    for times in timestamps_lists:
        time += times[1] - times[0]
    return time


def compute_percentage(array):
    return 100 * array / np.sum(array)


def compute_events_percentage(events_dataset):
    """
    Computes the average percentage of time spent in each state for each animal. It normalizes every state duration to
    the total length of the run, then takes the average across all the runs available for that condition.

    :param events_dataset: dataset with event timestamps
    :return: dataframe containing condition, animal and event containing the average time spent in each state
    """
    durations = events_dataset.copy()
    events_col = [col for col in durations.columns if col.startswith("event")]
    durations[events_col] = durations[events_col].applymap(compute_duration).apply(compute_percentage, axis=1)
    durations.drop("date", inplace=True, axis=1)
    means = durations.groupby(["condition", "subject"], as_index=False).mean(numeric_only=True)
    return means


def compute_ci(array):
    """
    Computes the confidence interval given hardcoded value of 1.96.

    :param array: array to compute the confidence interval on
    :return: confidence interval value
    """
    ci = 1.96 * np.std(array) / np.sqrt(len(array))
    return ci


def get_state_graph_stats(group_cond_df, stat="std"):
    """
    Takes the group dataset and returns a dataframe with mean, upper bound and lower bound to be used to graph.

    :param group_cond_df:
    :param stat: which statistic to use to compute the upper and lowe bounds. Takes "std" for standard deviation,
                 "sem" for standard error of mean and "ci" for confidence interval.
    :return: df
    """

    mean_df = group_cond_df.groupby(["group", "condition"], as_index=False).mean(numeric_only=True)
    if stat == "std":
        dist_df = group_cond_df.groupby(["group", "condition"], as_index=False).std(numeric_only=True)
    elif stat == "ci":
        dist_df = group_cond_df.groupby(["group", "condition"], as_index=False).apply(compute_ci, axis=0)
    elif stat == "sem":
        dist_df = group_cond_df.groupby(["group", "condition"], as_index=False).sem(numeric_only=True)
    else:
        raise ValueError(f"The statistic value {stat} is unsupported, please choose between std, sem or ci")

    mean_df = pd.melt(mean_df, id_vars=['group', 'condition'], value_vars=['event_gait', 'event_fog', 'event_nlm'],
                      var_name='event_type', value_name='mean')

    dist_df = pd.melt(dist_df, id_vars=['group', 'condition'], value_vars=['event_gait', 'event_fog', 'event_nlm'],
                      var_name='event_type', value_name='distribution')
    stats = mean_df.copy()
    stats["lower_bound"] = mean_df["mean"] - dist_df["distribution"]
    stats["upper_bound"] = mean_df["mean"] + dist_df["distribution"]

    return stats


def get_runs_list(experiment_structure, skip_subjects, skip_conditions):
    """
    Retrieves all the runs unique strings identifying day, subject, condition and run number

    :param experiment_structure: dictionary containing the experiment structure
    :param skip_subjects: which subjects to skip
    :param skip_conditions: which condition to skip
    :return: flat list with all the unique strings identifying runs
    """
    experiment_combos = []
    dates = [date for date in experiment_structure.keys()]
    for date in dates:
        subjects = [subject for subject in experiment_structure[date].keys() if subject not in skip_subjects]
        for subject in subjects:
            conditions = [condition for condition in experiment_structure[date][subject].keys() if
                          condition not in skip_conditions]
            for condition in conditions:
                runs = [str(run) for run in experiment_structure[date][subject][condition]]
                runs = ["0" + run if len(run) == 1 else run for run in runs]
                for run in runs:
                    experiment_combos.append([date, subject, condition, run])

    return experiment_combos
