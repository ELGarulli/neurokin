from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

import pandas as pd


class DefaultParams:
    def __init__(self, values: Dict[str, Any], types: Dict[str, List[type]]) -> None:
        self._assert_identical_keys(value_keys=values.keys(), type_keys=types.keys())
        self._assert_values_in_types_are_lists_of_types(types=types)
        self._assert_default_values_match_valid_types(values=values, types=types)
        self.values = values
        self.types = types

    def assert_input_params_and_fill_with_defaults(self, input_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if input_params is None:
            input_params = self.values
        else:
            assert (
                type(input_params) == dict
            ), f"input_params has to be a dictionary, not: {input_params}."
            for input_key, input_value in input_params.items():
                invalid_key_message = f'The key "{input_key}" of input_params does not match any of the valid keys: {self.values.keys()}'
                assert input_key in self.values.keys(), invalid_key_message
                invalid_value_type_message = (
                    f"input_params[{input_key}] has to be one of the following types: {self.types[input_key]}."
                    f"However, you passed: {input_value}, which is of type: {type(input_value)}"
                )
                assert (
                    type(input_value) in self.types[input_key]
                ), invalid_value_type_message
            for key, default_value in self.values.items():
                if key not in input_params:
                    input_params[key] = default_value
        return input_params

    def _assert_identical_keys(
        self, value_keys: List[str], type_keys: List[str]
    ) -> None:
        keys_dont_match_message = (
            'The keys of values & types have to be identical! However, "{}" '
            "of {} does not match with any key in {}: {}."
        )
        for key in value_keys:
            assert key in type_keys, keys_dont_match_message.format(
                key, "values", "types", type_keys
            )
        for key in type_keys:
            assert key in value_keys, keys_dont_match_message.format(
                key, "types", "values", value_keys
            )

    def _assert_values_in_types_are_lists_of_types(
        self, types: Dict[str, List[type]]
    ) -> None:
        for type_key, list_of_types in types.items():
            assert (
                type(list_of_types) == list
            ), f'The value of "{type_key}" in types is not a list: {list_of_types}.'
            not_all_elems_are_types_message = f""
            for elem in list_of_types:
                assert (
                    type(elem) == type
                ), f'The element "{elem}" in types[{type_key}] is not a type!'

    def _assert_default_values_match_valid_types(
        self, values: Dict[str, Any], types: Dict[str, List[type]]
    ) -> None:

        for key, default_value in values.items():
            invalid_default_value_type_message = (
                f'The default value for "{key}": {default_value}, is of '
                f"type: {type(default_value)}, which is not in the list "
                f"of valid types: {types[key]}."
            )
            assert type(default_value) in types[key], invalid_default_value_type_message


class FeatureExtraction(ABC):
    """
    Defines the interface for all feature extraction strategies available in neurokin.
    
    Takes a pd.DataFrame in DLC format (MultiIndex: ("scorer", "bodypart", "coords") 
    as input. It is recommended that this dataframe is already sliced to only those
    columns that are relevant for the feature extraction, hence: 'sliced_marker_df'.
    
    Optionally, a dictionary with parameter specifications can be passed. Since these 
    parameters are specified manually by the user in the config.yaml file, an instance 
    of the DefaultParams class is used to validate the user input, and to fill any 
    missing parameters with the corresponding default values.
    
    For developers:
    
    If you want to implement your own feature extraction strategy, please inherit from 
    this class & make sure to implement all abstractmethods. Check out their individual 
    docstrings for more information.
    """

    def __init__(self) -> None:
        self.default_params = self._initialize_default_params()

    @property
    @abstractmethod
    def input_type(self) -> str:
        """
        Please specify which type of input data is handled by this extraction strategy. E.g. "markers", "joints".
        """
        pass

    @property
    @abstractmethod
    def default_values(self) -> Dict[str, Any]:
        """
        If this feature extraction strategy allows the user to adjust any parameters, 
        please specify their default values here. For instance, if the user can 
        specify a parameter called "window_size" & you would like the default window 
        size to be 5, return:
        
        {'window_size': 5}
        
        If there are no adjustable parameters, simply return an empty dictionary.
        """
        pass

    @property
    @abstractmethod
    def default_value_types(self) -> Dict[str, List[type]]:
        """
        If this feature extraction strategy allows the user to adjust any parameters, 
        please specify the valid types of these adjustable parameters here.
        For instance, if the user can specify a parameter called "window_size" & you 
        require that this is an integer, return:
        
        {'window_size': [int]}
        
        Note, that the values of this dictionary have to be lists. This is to support 
        multiple valid types. For instance, if both floats & ints are valid, return:
        
        {'any_kind_of_number': [int, float]}
        
        Again, as for the "default_values" attribute: if there are no adjustable 
        parameters, simply return an empty dictionary.
        """
        pass

    @abstractmethod
    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: [Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        This is now where the magic of your feature extraction is supposed to happen. 
        The features shall be extracted for all marker IDs given in the list 
        "source_marker_ids".
        
        You can access the dataframe as "self.marker_df", and, if applicable, the 
        user-defined and already validated parameters as "self.params".
        
        !!! IMPORTANT !!!
        
        Please return a pd.DataFrame that contains ONLY the newly extracted features and 
        matches the number of rows of the original dataframe "self.marker_df". This will 
        be confirmed using the self._assert_valid_output() method.
        """
        pass

    def extract_features(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        params = self.default_params.assert_input_params_and_fill_with_defaults(
            input_params=params
        )

        extracted_features_df = self._run_feature_extraction(
            source_marker_ids=source_marker_ids, marker_df=marker_df, params=params
        )

        self._assert_valid_output(output_df=extracted_features_df, marker_df=marker_df)
        return extracted_features_df

    def _assert_valid_output(
        self, output_df: pd.DataFrame, marker_df: pd.DataFrame
    ) -> None:
        invalid_shape_message = (
            f"Rows of extracted features DataFrame ({output_df.shape[0]}) does not "
            f"match number of rows in the original DataFrame ({marker_df.shape[0]})."
        )
        assert output_df.shape[0] == marker_df.shape[0], invalid_shape_message

    def _initialize_default_params(self) -> DefaultParams:
        return DefaultParams(values=self.default_values, types=self.default_value_types)

    def _copy_filtered_columns_of_df(
        self,
        df_to_filter: pd.DataFrame,
        marker_id_filter: Union[slice, List[str], str],
        coords_filter: Union[slice, List[str], str] = slice(None),
        scorer_filter: Union[slice, List[str], str] = slice(None),
    ) -> pd.DataFrame:
        """
        Since these DataFrames with multi-indexed columns can be quite annoying to slice while 
        keeping the correct format (i.e. all three column levels "scorer", "bodyparts", "coords"), 
        this function might be relevant for many FeatureExtractionStrategy classes.
        """

        idx = pd.IndexSlice
        filtered_df = df_to_filter.loc[
            :, idx[scorer_filter, marker_id_filter, coords_filter]
        ].copy()
        return filtered_df

    def _rename_columns_on_selected_idx_level(
        self,
        df: pd.DataFrame,
        column_idx_level: Union[str, int] = 2,
        # refers to coords level; 'coords' would work the same
        prefix: str = "",
        suffix: str = "",
    ) -> pd.DataFrame:
        """
        Also renaming of columns only on a particular level (e.g. converting the coords 
        level from "x", "y", "z" to "x_sliding_mean", "y_sliding_mean", "z_sliding_mean") 
        while keeping all other levels (e.g. scorer & marker_ids) as-is, might be a function 
        of general in these classes and is therefore implemented here. 
        
        Be aware that this (and the underlying pandas function) assumes that the df always has 
        the same level values for all markers. For instance, if your input df has the marker
        columns: ["lshoulder", "rshoulder", "rmtp"], and "lshoulder" has three associated 
        columns: ["x", "y", "z"] on the "coords" level, it is assumed that these coords-columns
        are also present for "rshoulder" and "rmtp". See also: 
        
        https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.set_levels.html
        """
        current_column_names = list(
            df.columns.get_level_values(column_idx_level).unique()
        )
        new_column_names = [
            f"{prefix}{column_name}{suffix}" for column_name in current_column_names
        ]
        df.columns = df.columns.set_levels(new_column_names, level=column_idx_level)
        return df
