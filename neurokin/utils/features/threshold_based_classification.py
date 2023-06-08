import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from neurokin.utils.features.core import FeatureExtraction, DefaultParams


def _get_interval_border_idxs(
    all_matching_idxs: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Function written by Dennis Segebarth (DSegebarth on GitHub)
    Input: indices of a boolean series that are the same between bodyparts
    (e.g. all bodyparts are immobile/freezing)
    Output: list of tuples with start and end index of the intervals where all bodyparts
     are immobile/freezing
    """
    interval_border_idxs = []

    # check if there are any intervals
    if all_matching_idxs.shape[0] >= 1:
        # get start and end indices of intervals
        step_idxs = np.where(np.diff(all_matching_idxs) > 1)[0]
        step_end_idxs = np.concatenate(
            [step_idxs, np.array([all_matching_idxs.shape[0] - 1])]
        )
        step_start_idxs = np.concatenate([np.array([0]), step_idxs + 1])
        interval_start_idxs = all_matching_idxs[step_start_idxs]
        interval_end_idxs = all_matching_idxs[step_end_idxs]
        # loop over all intervals
        for start_idx, end_idx in zip(interval_start_idxs, interval_end_idxs):
            interval_border_idxs.append((start_idx, end_idx))
    return interval_border_idxs


class Immobility(FeatureExtraction):
    """
    Classifies bodyparts as immobile if the bodyparts speed is below a speed threshold
    Input:  df with positon data (i.e. DLC output) and speed for critical bodyparts
            source_marker_ids: List of markers for which immobility should be classified
    Additional information required in config file:
        immobility_threshold: float, speed threshold for immobility
        minimum_duration_immobility: float, minimum duration of immobility bouts
        fps: int, frames per second of recording
    Output: df with immobility bouts for each bodypart specified in source_marker_ids
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "immobility_threshold": 3,
            "markers_for_immobility": ["Snout", "TailBase"],
            "minimum_duration_immobility": 0.1,
            "fps": 80,
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "immobility_threshold": int,
            "markers_for_immobility": [str],
            "minimum_duration_immobility": int,
            "fps": int,
        }
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # Check whether immobility is already calculated
        if ("scorer", source_marker_ids, "immobility") in marker_df.columns:
            pass
        else:
            # Check whether speed is calculated for all markers critical for immobility
            if ("scorer", source_marker_ids, "speed") in marker_df.columns:
                filtered_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=source_marker_ids,
                    coords_filter=["speed"],
                )

                # remove multiindex to make it easier to work with
                filtered_df = filtered_df.droplevel([0, 1], axis=1)

                # create new col for immobility, set val to false
                filtered_df["immobility"] = False
                # filter for indices in which the speed is less than the immobility threshold
                immobility_candidate_bout_idxs = np.where(
                    filtered_df["speed"] < params["immobility_threshold"]
                )[0]

                immobility_candidate_border_idxs = _get_interval_border_idxs(
                    all_matching_idxs=immobility_candidate_bout_idxs
                )

                # iterate through idxs
                # set immobility to true, determine duration of immobility bout
                for start_idx, end_idx in immobility_candidate_border_idxs:
                    interval_duration = (end_idx - start_idx + 1) * (1 / params["fps"])
                    if interval_duration >= params["minimum_duration_immobility"]:
                        filtered_df.loc[start_idx:end_idx, "immobility"] = True
                        filtered_df.loc[
                            start_idx:end_idx, "immobility_bout_duration"
                        ] = interval_duration

                # transform dfs into multiindex df

                # immobility column (boolean)
                immobility_col_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart=source_marker_ids,
                    axis="immobility",
                    data=filtered_df.loc[:, "immobility"],
                )

                # immobility bout duration
                immobility_bout_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart=source_marker_ids,
                    axis="immobility_bout_duration",
                    data=filtered_df.loc[:, "immobility_bout_duration"],
                )

                final_df = pd.concat([immobility_col_df, immobility_bout_df], axis=1)

                return final_df


class Freezing(FeatureExtraction):
    """
    Classifies subject as freezing if the necessary markers are immobile
    Input:  df with positon data (i.e. DLC output) and immobility for critical bodyparts
            source_marker_ids: List of markers, not used in the function but needs to get passed
    Additional information required in config file:
        markers_for_freezing: markers used for immobility detection
        minimum_duration_freezing: float, minimum duration of freezing bouts
        fps: int, frames per second of recording
    Output: df with freezing bouts for subject
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "markers_for_freezing": ["Snout", "TailBase"],
            "front_marker": "Snout",
            "minimum_duration_freezing": 0.5,
            "fps": 80,
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "markers_for_freezing": [str],
            "front_marker": str,
            "minimum_duration_freezing": int,
            "fps": int,
        }
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # Check whether freezing is already calculated
        if ("scorer", "subject", "freezing") in marker_df.columns:
            pass
        else:
            # Check whether speed is calculated for all markers critical for immobility
            if all(
                ("scorer", column, "immobility") in marker_df.columns
                for column in params["markers_for_freezing"]
            ):
                filtered_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=params["markers_for_freezing"],
                    coords_filter=["immobility", "immobility_bout_duration"],
                )
                # drop scorer level
                filtered_df = filtered_df.droplevel([0], axis=1)

                # get idxs of freezing bouts -> idxs of where all markers are immobile
                # and immobility bout duration is >= minimum_duration_freezing
                all_freezing_bout_idxs = []
                for marker in params["markers_for_freezing"]:
                    filtered_df.loc[
                        filtered_df[marker]["immobility_bout_duration"]
                        >= params["minimum_duration_freezing"],
                        "freezing",
                    ] = True

                    freezing_bout_idxs_per_marker = np.where(
                        filtered_df[marker]["immobility"] == True
                    )[0]

                    all_freezing_bout_idxs.append(freezing_bout_idxs_per_marker)

                shared_valid_idxs_for_all_markers = all_freezing_bout_idxs[0]
                if len(all_freezing_bout_idxs) > 1:
                    for next_set_of_valid_idxs in all_freezing_bout_idxs[1:]:
                        shared_valid_idxs_for_all_markers = np.intersect1d(
                            shared_valid_idxs_for_all_markers, next_set_of_valid_idxs
                        )

                # drop bodypart level
                filtered_df.droplevel([0], axis=1)

                # get freezing bout start and end idxs
                freezing_bout_border_idxs = _get_interval_border_idxs(
                    shared_valid_idxs_for_all_markers
                )

                # create new col for freezing, set val to false
                filtered_df["freezing"] = False

                # get x position of front marker -> place of freezing
                x_position_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=[params["front_marker"]],
                    coords_filter=["x"],
                )
                x_position_df = x_position_df.droplevel([0, 1], axis=1)

                # iterate through idxs
                # set gait disruption to true, determine duration, x position, and bout nr
                bout_nr = 1
                for start_idx, end_idx in freezing_bout_border_idxs:
                    interval_duration = (end_idx - start_idx + 1) / params["fps"]
                    # since the interval min duration is checked per marker and then the idxs are intersected it is
                    # necessary to check the interval duration again
                    if interval_duration >= params["minimum_duration_freezing"]:
                        filtered_df.loc[start_idx:end_idx, "freezing"] = True
                        filtered_df.loc[
                            start_idx:end_idx, "freezing_bout_duration"
                        ] = interval_duration
                        filtered_df.loc[start_idx:end_idx, "freezing_bout_start"] = (
                            start_idx * 1 / params["fps"]
                        )
                        filtered_df.loc[start_idx:end_idx, "freezing_bout_end"] = (
                            end_idx * 1 / params["fps"]
                        )
                        filtered_df.loc[
                            start_idx:end_idx, "freezing_bout_duration"
                        ] = interval_duration
                        x_position = x_position_df.loc[start_idx, "x"]
                        filtered_df.loc[
                            start_idx:end_idx, "freezing_bout_x_position"
                        ] = x_position
                        filtered_df.loc[start_idx:end_idx, "freezing_bout_nr"] = bout_nr
                        bout_nr += 1
                    else:
                        continue

                # transform dfs into multiindex df
                # freezing
                freezing_col_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="freezing",
                    data=filtered_df.loc[:, "freezing"],
                )
                # check if there are any freezing bouts
                if filtered_df.loc[filtered_df["freezing"] == True].empty:
                    print('no freezing bouts were detected with the given parameters')
                    return freezing_col_df
                else:
                    freezing_start_df = self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="freezing_bout_start",
                        data=filtered_df.loc[:, "freezing_bout_start"],
                    )
                    freezing_end_df = self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="freezing_bout_end",
                        data=filtered_df.loc[:, "freezing_bout_end"],
                    )

                    freezing_bout_duration_df = (
                        self.convert_singleindex_to_multiindex_df(
                            scorer="scorer",
                            bodypart="subject",
                            axis="freezing_bout_duration",
                            data=filtered_df.loc[:, "freezing_bout_duration"],
                        )
                    )
                    freezing_bout_x_position_df = (
                        self.convert_singleindex_to_multiindex_df(
                            scorer="scorer",
                            bodypart="subject",
                            axis="freezing_bout_x_position",
                            data=filtered_df.loc[:, "freezing_bout_x_position"],
                        )
                    )
                    freezing_bout_nr_df = self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="freezing_bout_number",
                        data=filtered_df.loc[:, "freezing_bout_nr"],
                    )
                    final_df = pd.concat(
                        [
                            freezing_col_df,
                            freezing_start_df,
                            freezing_end_df,
                            freezing_bout_duration_df,
                            freezing_bout_x_position_df,
                            freezing_bout_nr_df,
                        ],
                        axis=1,
                    )
                    return final_df

            else:
                raise ValueError(
                    "Be sure to classify immobility for each bodypart before classifying freezing!"
                )


class GaitDisruption(FeatureExtraction):
    """
    Classifies movement per marker for each marker in markers_for_gait_disruption
    Classifies gait disruption bouts based on immobility bouts and movement for each marker in markers_for_gait_disruption
    Checks overlap of gait disruption bouts for each marker in markers_for_gait_disruption
    Using the shared indices of gait disruption, classifies gait disruption bouts for the whole subject
    Adds information about bout status (bool) bout duration, x_position, and bout number to df

    Input:  df with positon data (i.e. DLC output) and immobility detection
            source_marker_ids: list of marker ids
    Additional information required in config file:
        min_duration_movement_before: float, minimal duration of movement before gaits disruption bout
        min_duration_immobility: float, minimal duration of immobility during gait disruption bout
        front_of_subject: str, marker id of marker in front of subject
        fps: float, frames per second
    Output: df with gait disruption bouts (bool)
            gait disruption duration and
            gait disruption position (x) for the whole subject
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "markers_for_gait_disruption": ["TailBase"],
            "min_duration_movement_before": 0.5,
            "min_duration_immobility": 0.1,
            "front_of_subject": "Snout",
            "fps": 80,
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "markers_for_gait_disruption": List[str],
            "min_duration_movement_before": float,
            "min_duration_immobility": float,
            "front_of_subject": str,
            "fps": float,
        }
        return default_types

    #
    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # Check whether gait disruption is already calculated
        if ("scorer", "subject", "gait_disruption") in marker_df.columns:
            pass
        else:
            # check whether immobility is calculated (necessary for gait disruption classification)
            if all(
                ("scorer", column, "immobility") in marker_df.columns
                for column in params["markers_for_gait_disruption"]
            ):
                # slice marker df
                filtered_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=params["markers_for_gait_disruption"],
                    coords_filter=["immobility", "immobility_bout_duration"],
                )

                # remove multiindex to make it easier to work with
                filtered_df = filtered_df.droplevel([0], axis=1)

                all_gait_disruption_bout_candidate_idxs = []
                for marker in params["markers_for_gait_disruption"]:
                    # create temporary df for each marker to seperate it clearly from df that is used lated
                    temp_df = filtered_df[marker].copy()

                    # classify movement
                    movement_interval_start_idxs = np.where(
                        (temp_df["immobility"] == False)
                        & (temp_df["immobility"].shift(1) == True)
                    )[0]
                    # add first index if first frame is not immobile
                    if temp_df["immobility"][0] == False:
                        movement_interval_start_idxs = np.append(
                            arr=movement_interval_start_idxs, values=0
                        )
                        movement_interval_start_idxs = np.sort(
                            movement_interval_start_idxs
                        )

                    movement_interval_end_idxs = np.where(
                        (temp_df["immobility"] == False)
                        & (temp_df["immobility"].shift(-1) == True)
                    )[0]

                    for start_idx, end_idx in zip(
                        movement_interval_start_idxs, movement_interval_end_idxs
                    ):
                        movement_interval_frame_count = (end_idx + 1) - start_idx
                        movement_interval_duration = (
                            movement_interval_frame_count * 1 / params["fps"]
                        )
                        temp_df.loc[
                            start_idx:end_idx, "movement_bout_duration"
                        ] = movement_interval_duration

                    gait_disruption_start_idxs = np.where(
                        (
                            temp_df["immobility_bout_duration"]
                            >= params["min_duration_immobility"]
                        )
                        & (
                            temp_df["movement_bout_duration"].shift(1)
                            >= params["min_duration_movement_before"]
                        )
                    )[0]

                    for idx in gait_disruption_start_idxs:
                        gait_disruption_bout_duration_frames_per_marker = (
                            temp_df["immobility_bout_duration"].loc[idx] * params["fps"]
                        )

                        temp_df.loc[
                            idx : idx + gait_disruption_bout_duration_frames_per_marker,
                            "gait_disruption_per_marker",
                        ] = True

                    gait_disruption_bout_idxs_per_marker = np.where(
                        temp_df["gait_disruption_per_marker"] == True
                    )[0]
                    all_gait_disruption_bout_candidate_idxs.append(
                        gait_disruption_bout_idxs_per_marker
                    )

                # drop level 0 of multiindex (bodyparts)
                filtered_df = filtered_df.droplevel([0], axis=1)

                # get shared valid idxs for all markers
                shared_valid_idxs_for_all_markers = (
                    all_gait_disruption_bout_candidate_idxs[0]
                )
                if len(all_gait_disruption_bout_candidate_idxs) > 1:
                    for (
                        next_set_of_valid_idxs
                    ) in all_gait_disruption_bout_candidate_idxs[1:]:
                        shared_valid_idxs_for_all_markers = np.intersect1d(
                            shared_valid_idxs_for_all_markers, next_set_of_valid_idxs
                        )

                # get border idxs of gait disruption bouts
                gait_disruption_bout_border_idxs = _get_interval_border_idxs(
                    shared_valid_idxs_for_all_markers
                )

                # get x position of front marker of subject -> position of bout
                x_position_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=[params["front_marker"]],
                    coords_filter=["x"],
                )
                x_position_df = x_position_df.droplevel([0, 1], axis=1)

                # create new col for gait_disruption, set val to false
                filtered_df["gait_disruption"] = False

                # set gait disruption to true, determine duration, x position, and bout nr
                bout_nr = 1

                # iterate through shared border idxs of gait disruption bouts
                for start_idx, end_idx in gait_disruption_bout_border_idxs:
                    interval_duration = (end_idx + 1 - start_idx) * 1 / params["fps"]
                    # since the interval min duration is checked per marker and then the idxs are intersected it is
                    # necessary to check the interval duration again
                    if interval_duration >= params["min_duration_immobility"]:
                        filtered_df.loc[start_idx:end_idx, "gait_disruption"] = True
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_duration"
                        ] = interval_duration
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_start"
                        ] = (start_idx * 1 / params["fps"])
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_end"
                        ] = (end_idx * 1 / params["fps"])
                        interval_duration = (
                            (end_idx + 1 - start_idx) * 1 / params["fps"]
                        )
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_duration"
                        ] = interval_duration
                        x_position = x_position_df.loc[start_idx, "x"]
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_x_position"
                        ] = x_position
                        filtered_df.loc[
                            start_idx:end_idx, "gait_disruption_bout_nr"
                        ] = bout_nr
                        bout_nr += 1
                    else:
                        continue

                # transform dfs into multiindex df
                # gait disruption
                gait_disruption_col_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="gait_disruption",
                    data=filtered_df.loc[:, "gait_disruption"],
                )
                # gait disruption bout start
                gait_disruption_start_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="gait_disruption_bout_start",
                    data=filtered_df.loc[:, "gait_disruption_bout_start"],
                )
                # gait disruption bout end
                gait_disruption_end_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="gait_disruption_bout_end",
                    data=filtered_df.loc[:, "gait_disruption_bout_end"],
                )
                # gait disruption bout duration
                gait_disruption_bout_duration_df = (
                    self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="gait_disruption_bout_duration",
                        data=filtered_df.loc[:, "gait_disruption_bout_duration"],
                    )
                )
                # gait disruption bout x position
                gait_disruption_bout_x_position_df = (
                    self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="gait_disruption_bout_x_position",
                        data=filtered_df.loc[:, "gait_disruption_bout_x_position"],
                    )
                )
                # gait disruption bout number
                gait_disruption_bout_nr_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="gait_disruption_bout_number",
                    data=filtered_df.loc[:, "gait_disruption_bout_nr"],
                )

                final_df = pd.concat(
                    [
                        gait_disruption_col_df,
                        gait_disruption_start_df,
                        gait_disruption_end_df,
                        gait_disruption_bout_duration_df,
                        gait_disruption_bout_x_position_df,
                        gait_disruption_bout_nr_df,
                    ],
                    axis=1,
                )
                return final_df

            else:
                raise ValueError(
                    "Be sure to classify immobility before classifying gait disruption!"
                )
