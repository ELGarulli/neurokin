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
                cond_animal_event_dict[condition].setdefault(animal, {"gait": [],
                                                                      "fog_active": [],
                                                                      "fog_rest": [],
                                                                      "nlm_active": [],
                                                                      "nlm_rest": []})
                runs = list(psds_correlates_dict[condition][date][animal])
                for run in runs:
                    for fog in conditions_of_interest[date][animal][run]["fog_active"]:
                        if len(fog) > 0:
                            cond_animal_event_dict[animal]["fog_active"].append(fog)

                    for fog in conditions_of_interest[date][animal][run]["fog_rest"]:
                        if len(fog) > 0:
                            cond_animal_event_dict[animal]["fog_rest"].append(fog)

                    for gait in conditions_of_interest[date][animal][run]["gait"]:
                        if len(gait) > 0:
                            cond_animal_event_dict[animal]["gait"].append(gait)

                    for interr in conditions_of_interest[date][animal][run]["nlm_active"]:
                        if len(interr) > 0:
                            cond_animal_event_dict[animal]["nlm_active"].append(interr)

                    for interr in conditions_of_interest[date][animal][run]["nlm_rest"]:
                        if len(interr) > 0:
                            cond_animal_event_dict[animal]["nlm_rest"].append(interr)
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
                    five2three[animal].setdefault("fog", [])
                    five2three[animal]["fog"].append(psds)
                elif event_type in ["nlm_active", "nlm_rest"]:
                    five2three[animal].setdefault("nlm", [])
                    five2three[animal]["nlm"].append(psds)
                else:
                    five2three[animal].setdefault("gait", [])
                    five2three[animal]["gait"].append(psds)
    return five2three


def get_per_animal_average(cond_animal_event_dict):
    conditions_of_interest = list(cond_animal_event_dict.keys())
    animals_avg_psds = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, events in cond_animal_event_dict[condition].items():
            for event, psds in events.items():
                animals_avg_psds[condition].setdefault(animal, {})
                animals_avg_psds[condition][animal][event] = np.mean(psds, axis=0)
    return animals_avg_psds


def get_group_split(model_list, animals_avg_psds):
    conditions_of_interest = list(animals_avg_psds.keys())
    model_group = {c: {} for c in conditions_of_interest}
    sham_group = {c: {} for c in conditions_of_interest}
    for condition in conditions_of_interest:
        for animal, event in animals_avg_psds.items():
            for event_name, event_array in event.items():
                if animal in model_list:
                    model_group[condition].setdefault(animal, {})
                    model_group[condition][animal][event_name] = event_array
                else:
                    sham_group[condition].setdefault(animal, {})
                    sham_group[condition][animal][event_name] = event_array

    groups_psds_split = {"model_group": model_group,
                         "sham_group": sham_group}
    return groups_psds_split


def get_group_average(model_list, groups_psds_split):
    split_group = get_group_split(model_list, groups_psds_split)
    groups = list(split_group.keys())
    groups_avg = {}

    for group in groups:
        groups_avg.setdefault(group, {})
        for condition in list(groups.groups_psds_split[group]()):
            groups_avg[group].setdefault(condition, {})
            condition_average = {event: [] for event in groups_psds_split[group][condition]}
            for animal, event in groups_psds_split.items():
                for event_name, event_array in event.items():
                    condition_average[event_name].append(event_array)
            groups_avg[group][condition] = condition_average

    return groups_avg
