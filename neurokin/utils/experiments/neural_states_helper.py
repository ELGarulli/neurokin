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


# PANDIZE
def get_events_per_animal(psds_correlates_dict):
    """
    Simplifies the dataset, merging all the dates and run levels. Returning a dictionary that contains the stimulation
    categories, the animals ID and the event type.
    :param psds_correlates_dict: psds correlates dictionary structured as condition, date, animal, run, events.
    :return:
    """
    conditions_of_interest = list(psds_correlates_dict.keys())
    cond_animal_event_dict = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        dates = list(psds_correlates_dict[condition].keys())
        for date in dates:
            animals = list(psds_correlates_dict[condition][date].keys())
            for animal in animals:
                cond_animal_event_dict[condition].setdefault(animal, {})
                runs = list(psds_correlates_dict[condition][date][animal].keys())
                for run in runs:
                    for event_name, event_array in psds_correlates_dict[condition][date][animal][run].items():
                        cond_animal_event_dict[condition][animal].setdefault(event_name, [])
                        if len(event_array) > 0:
                            cond_animal_event_dict[condition][animal][event_name] += event_array
    return cond_animal_event_dict


# PANDIZE
def condense_neural_event_types(cond_animal_event_dict):
    """
    Condenses the dataset from the five categories analysis
    ["fog_active", "fog_rest", "nlm_active", "nlm_rest", gait] to only
    three categories ["fog", "nlm", "gait"]
    :param cond_animal_event_dict: states dictionary structured as condition, animal, events.
    :return: same dictionary with 3 conditions instead of 5
    """
    conditions_of_interest = list(cond_animal_event_dict.keys())
    five2three = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, events in cond_animal_event_dict[condition].items():
            five2three[condition][animal] = {}
            for event_type, psds in events.items():
                if event_type in ["fog_active", "fog_rest"]:
                    five2three[condition][animal].setdefault("fog", [])
                    five2three[condition][animal]["fog"] += psds
                elif event_type in ["nlm_active", "nlm_rest"]:
                    five2three[condition][animal].setdefault("nlm", [])
                    five2three[condition][animal]["nlm"] += psds
                else:
                    five2three[condition][animal].setdefault("gait", [])
                    five2three[condition][animal]["gait"] += psds
    return five2three


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


# PANDIZE
def _condense_distribution_event_types(cond_animal_event_dict):
    """
    Condenses from the dataset five categories analysis
    ["fog_active", "fog_rest", "nlm_active", "nlm_rest", gait] to only
    three categories ["fog", "nlm", "gait"]
    :param cond_animal_event_dict: states dictionary structured as condition, animal, events.
    :return:  same dictionary with 3 conditions instead of 5
    """
    conditions_of_interest = list(cond_animal_event_dict.keys())
    five2three = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, events in cond_animal_event_dict[condition].items():
            five2three[condition][animal] = {}
            for event_type, psds in events.items():
                if event_type in ["fog_active", "fog_rest"]:
                    five2three[condition][animal].setdefault("fog", 0)
                    five2three[condition][animal]["fog"] += psds
                elif event_type in ["nlm_active", "nlm_rest"]:
                    five2three[condition][animal].setdefault("nlm", 0)
                    five2three[condition][animal]["nlm"] += psds
                else:
                    five2three[condition][animal].setdefault("gait", 0)
                    five2three[condition][animal]["gait"] += psds
    return five2three


# PANDIZE
def get_per_animal_average(cond_animal_event_dict):
    """
    Computes the average psds response per event per animal.
    :param cond_animal_event_dict:
    :return: dictionary structured as condition, animal, event, the final layer contains the average psds of that event
    """
    conditions_of_interest = list(cond_animal_event_dict.keys())
    animals_avg_psds = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, events in cond_animal_event_dict[condition].items():
            for event, psds in events.items():
                animals_avg_psds[condition].setdefault(animal, {})
                psds = [psd for psd in psds if not np.isnan(psd).any()]
                average_event = np.mean(psds, axis=0)  # FIXME why is this calculated twice
                if np.isnan(average_event).any():
                    print(f"Attention! No valid events was found for condition: {condition}, animal: {animal}, "
                          f"event type: {event}.")
                else:
                    animals_avg_psds[condition][animal][event] = np.mean(psds, axis=0)
    return animals_avg_psds


def get_group_split(test_sbj_list, percentage_df):
    """
    Splits the dataset in two groups depending on which subjects are in the test group and which not (control)
    :param test_sbj_list: list of subject IDs that belong to teh test group
    :param animals_avg_dataset: dictionary containing the psds organized by condition, animal and event
    :return: split dictionary between test_group and sham_group
    """
    subject_groups = percentage_df.copy()
    subject_groups["group"] = subject_groups["subject"].apply(lambda x: x in test_sbj_list)
    return subject_groups


# PANDIZE
def _get_group_split(test_sbj_list, animals_avg_dataset):
    """
    Splits the dataset in two groups depending on which subjects are in the test group and which not (control)
    :param test_sbj_list: list of subject IDs that belong to teh test group
    :param animals_avg_dataset: dictionary containing the psds organized by condition, animal and event
    :return: split dictionary between test_group and sham_group
    """
    conditions_of_interest = list(animals_avg_dataset.keys())
    model_group = {c: {} for c in conditions_of_interest}
    sham_group = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, event in animals_avg_dataset[condition].items():
            if animal in test_sbj_list:
                for event_name, event_array in event.items():
                    model_group[condition].setdefault(animal, {})
                    model_group[condition][animal][event_name] = event_array
            else:
                for event_name, event_array in event.items():
                    sham_group[condition].setdefault(animal, {})
                    sham_group[condition][animal][event_name] = event_array

    groups_split = {"test_group": model_group,
                    "sham_group": sham_group}
    return groups_split


# PANDIZE
def drop_subject_id(test_sbj_list, animals_avg_psds):
    """
    First it splits the dictionary in test and sham group. Then in each group it runs through the categories,
    animals and events and merges averaged data coming from animals in the same experiment group
    :param test_sbj_list: list of subject IDs that belong to the test group
    :param animals_avg_psds: dictionary structured as condition, animal, event,
                            the final layer contains the average psds of that event
    :return: simplified dictionary where the animals ID are dropped, maintaining a split between test and sham group
    """
    split_group = get_group_split(test_sbj_list, animals_avg_psds)
    groups = list(split_group.keys())
    merged_animal_dict = {groups: {} for groups in groups}
    groups_avg = {groups: {} for groups in groups}

    for group in groups:
        conditions = list(split_group[group].keys())
        merged_animal_dict[group] = {condition: {} for condition in conditions}
        for condition in conditions:
            groups_avg[group].setdefault(condition, {})
            for animal, events_dict in split_group[group][condition].items():
                for event_name, event_array in events_dict.items():
                    merged_animal_dict[group][condition].setdefault(event_name, [])
                    merged_animal_dict[group][condition][event_name] += [event_array]
            for event_name, events_arrays in merged_animal_dict[group][condition].items():
                groups_avg[group][condition][event_name] = events_arrays

    return groups_avg


# PANDIZE
def get_group_average(test_sbj_list, animals_avg_psds):
    """
    Given the dictionary containing the psds organized by condition, animal and event, it splits it in two groups and
    computes the average per event
    :param test_sbj_list: list of subject IDs that belong to the test group
    :param animals_avg_psds: dictionary containing the psds organized by condition, animal and event
    :return: dictionary organized by experiment group, containing the group average per event per condition
    """

    no_subject_id_dataset = drop_subject_id(test_sbj_list, animals_avg_psds)
    groups = list(no_subject_id_dataset.keys())
    groups_avg = {groups: {} for groups in groups}

    for group in groups:
        conditions = list(no_subject_id_dataset[group].keys())
        for condition in conditions:
            groups_avg[group].setdefault(condition, {})
            for event_name, event_array in no_subject_id_dataset[group][condition].items():
                groups_avg[group][condition][event_name] = np.mean(event_array, axis=0)

    return groups_avg


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

# PANDIZE
def compute_state_distribution_stats(group_cond_dataset, stat="std"):
    """
    Takes a dictionary structured as group, condition, animal and event containing the average
    time spent in each state, and computes the mean, upper bound and lower bound for each state,
    keeping the separation in groups and conditions.
    :param group_cond_dataset: dictionary structured as group, condition, animal and event
                                containing the average time spent in each state
    :param stat: which statistic to use to compute the upper and lowe bounds. Takes "std" for standard deviation,
                 "sem" for standard error of mean and "ci" for confidence interval.
    :return: dictionary organized as group, condition, state. The keys to access the stats are
            "mean", "upper_bound", "lower_bound"
    """
    stats_dict = {}
    for group, condition_dict in group_cond_dataset.items():
        stats_dict.setdefault(group, {})
        for condition, states_dict in condition_dict.items():
            stats_dict[group].setdefault(condition, {})
            condition_dataframe = pd.DataFrame(states_dict).T
            states = list(condition_dataframe.columns)
            for state in states:
                mean = condition_dataframe[state].mean()
                if stat == "std":
                    std = condition_dataframe[state].std()
                    positive_stat = mean + std
                    negative_stat = mean - std
                if stat == "ci":
                    ci = 1.96 * condition_dataframe[state].std() / np.sqrt(len(condition_dataframe[state]))
                    positive_stat = mean + ci
                    negative_stat = mean - ci
                if stat == "sem":
                    sem = condition_dataframe[state].sem()
                    positive_stat = mean + sem
                    negative_stat = mean - sem
                stats_dict[group][condition][state] = {"mean": mean,
                                                       "upper_bound": positive_stat,
                                                       "lower_bound": negative_stat}
    return stats_dict


def get_runs_list(experiment_structure, skip_subjects, skip_conditions):
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
