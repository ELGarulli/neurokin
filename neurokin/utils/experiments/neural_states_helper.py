import numpy as np


def get_events_per_animal(psds_correlates_dict):
    """
    Simplifies the dataset, merging all the date and run levels. Returning a dictionary that contatins the stimulation
    categories, the animals ID and the event type.
    :param psds_correlates_dict:
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


def condense_event_types(cond_animal_event_dict):
    """
    Condenses from the five categories analysis ["fog_active", "fog_rest", "nlm_active", "nlm_rest", gait] to only
    three categories ["fog", "nlm", "gait"]
    :return:
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


def get_per_animal_average(cond_animal_event_dict):
    """
    Computes the average psds response per event per animal.
    :param cond_animal_event_dict:
    :return:
    """
    conditions_of_interest = list(cond_animal_event_dict.keys())
    animals_avg_psds = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, events in cond_animal_event_dict[condition].items():
            for event, psds in events.items():
                animals_avg_psds[condition].setdefault(animal, {})
                average_event = np.mean(psds, axis=0)
                if np.isnan(average_event).any():
                    print(f"Attention! No valid events was found for animal: {animal}, "
                          f"event type: {event}.")
                else:
                    animals_avg_psds[condition][animal][event] = np.mean(psds, axis=0)
    return animals_avg_psds


def get_group_split(test_sbj_list, animals_avg_psds):
    """
    Splits the dataset in two groups depending on which subjects are in the test group and which not (control)
    :param test_sbj_list: list of subject IDs that belong to teh test group
    :param animals_avg_psds: dictionary containing the psds organized by condition, animal and event
    :return:
    """
    conditions_of_interest = list(animals_avg_psds.keys())
    model_group = {c: {} for c in conditions_of_interest}
    sham_group = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, event in animals_avg_psds[condition].items():
            if animal in test_sbj_list:
                for event_name, event_array in event.items():
                    model_group[condition].setdefault(animal, {})
                    model_group[condition][animal][event_name] = event_array
            else:
                for event_name, event_array in event.items():
                    sham_group[condition].setdefault(animal, {})
                    sham_group[condition][animal][event_name] = event_array

    groups_psds_split = {"model_group": model_group,
                         "sham_group": sham_group}
    return groups_psds_split


def drop_subject_id(test_sbj_list, animals_avg_psds):
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
