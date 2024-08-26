import numpy as np
from neurokin.utils.neural import processing
from neurokin.neural_data import NeuralData
import pytest


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


class TestSimplyMeanDataBinarize:

    def test_simply_mean_data_binarize_with_empty_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([])), np.array([]))

    def test_simply_mean_data_binarize_with_1_element_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([1.0])), np.array([1]))

    def test_simply_mean_data_binarize_with_random_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([1.0, 2.0, 3.0, 4.0])),
                              np.array([0, 0, 1, 1]))

    @pytest.mark.skip(reason="fails due to nan comparisons")
    def test_simply_mean_data_binarize_with_nan_array(self):
        # assert np.array_equal(processing.simply_mean_data_binarize(np.array([1, np.nan])), np.array([1, 1]))
        with pytest.raises(ValueError):
            processing.simply_mean_data_binarize(np.array([1, np.nan]))  # currently returns np.array([1, 1])

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_simply_mean_data_binarize_with_2d_array(self):
        # with pytest.raises(ValueError):
        #    processing.simply_mean_data_binarize(np.array([[1, 1, 2], [1, 2, 3]]))  # should be np.array([[0, 0, 1],[0, 1, 1]])
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([[1, 1, 2], [1, 2, 3]])),
                              np.array([[0, 0, 1], [0, 1, 1]]))


class TestGetStimTimestamps:

    def test_get_stim_timestamps_with_empty_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([])), np.array([]))

    def test_get_stim_timestamps_with_nan_array(self):  # should this return Warning or ValueError?
        assert np.array_equal(processing.get_stim_timestamps(np.array([np.nan])), np.array([]))
        assert np.array_equal(processing.get_stim_timestamps(np.array([np.nan, 1])), np.array([1]))

    @pytest.mark.skip(reason="fails due to lack of check for dim")
    def test_get_stim_timestamps_with_2d_array(self):
        with pytest.raises(ValueError):  # currently returns np.array([0])
            processing.get_stim_timestamps(np.array([[1, 2, 3], [0, 2, 3]]))

    def test_get_stim_timestamps_with_random_array_without_expected_pulses(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_greater_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=3),
            np.array([1, 8]))  # should there be a Warning message here

    def test_get_stim_timestamps_with_random_array_with_exact_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=2),
            np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_lower_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=1),
            np.array([1]))

    def test_get_stim_timestamps_with_random_array_with_0_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=0),
            np.array([1, 8]))  # should this return empty array np.array([])?

    @pytest.mark.skip(reason="fails due to no handling of negative expected_pulses")
    def test_get_stim_timestamps_with_random_array_with_negative_expected_pulses(self):
        #assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-1),
        #    np.array([1]))
        #assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-2),
        #    np.array([]))  # might want to return ValueError?
        #assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-3),
        #    np.array([]))  # might want to return ValueError?
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-1)
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-2)

    def test_get_stim_timestamps_with_random_array_mid_pulse(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([1, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([0, 8]))

    def test_get_stim_timestamps_with_0_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 0, 0, 0, 0, 0])), np.array([]))


class TestGetTTimestampsStimBlocks:

    @pytest.fixture
    def neural_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.get_stim_timestamps', side_effect=self.get_stim_timestamps)
        mocker.patch('neurokin.utils.neural.importing.time_to_sample', side_effect=self.time_to_sample)
        data = NeuralData("")
        data.fs = 1
        data.sync_data = np.array([0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0])
        yield data

    def get_stim_timestamps(self, sync_ch: np.ndarray, expected_pulses: int = None) -> np.ndarray:
        key = get_key(str(sync_ch))
        return_values = {get_key(str(np.array([]))): {str(1): np.array([])},
                         get_key(str(np.array([0, 0, 0, 0, 0, 0]))): {str(1): np.array([])},
                         get_key(str(np.array([0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0]))): {
                             str(6): np.array([1, 3, 6, 8, 11, 13]),
                             str(4): np.array([1, 3, 6, 8]),
                             str(8): np.array([1, 3, 6, 8, 11, 13]),
                             str(0): np.array([1, 3, 6, 8, 11, 13])},
                         get_key(str(np.array([0, 1, 1, 0, 1, 1, 0, 0, 2, 2, 0, 2, 2, 0, 0, 3, 3, 0, 3]))): {
                             str(6): np.array([1, 4, 8, 11, 15, 18])}
                         }
        return return_values.get(key, "default_output")[str(expected_pulses)]

    def time_to_sample(self, timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
        return_values = {str(3): 2,
                         str(1): 0,
                         str(5): 4}
        return return_values[str(timestamp)]

    def test_get_timestamps_stim_blocks_with_empty_array(self, neural_data):
        neural_data.sync_data = np.array([])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=1, pulses=1, time_stim=1) == []

    def test_get_timestamps_stim_blocks_with_0_array(self, neural_data):
        neural_data.sync_data = np.array([0, 0, 0, 0, 0, 0])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=1, pulses=1, time_stim=1) == []

    def test_get_timestamps_stim_blocks_with_random_array_and_exact_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=3, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_and_lower_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=2, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8)]

    def test_get_timestamps_stim_blocks_with_random_array_and_greater_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=4, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_and_0_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=0, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_with_1_pulse_per_block(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=6, pulses=1, time_stim=1) == [(1, 1),
                                                                                                             (3, 3),
                                                                                                             (6, 6),
                                                                                                             (8, 8),
                                                                                                             (11, 11),
                                                                                                             (13, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_ending_mid_pulse(self,
                                                                           neural_data):  # should the end of the last block be adjusted to the length of the array?
        neural_data.sync_data = np.array([0, 1, 1, 0, 1, 1, 0, 0, 2, 2, 0, 2, 2, 0, 0, 3, 3, 0, 3])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=3, pulses=2, time_stim=5) == [(1, 5),
                                                                                                             (8, 12),
                                                                                                             (15, 19)]


class TestGetMedianDistance:

    def test_get_median_distance_with_empty_array(self):
        assert np.isnan(processing.get_median_distance(np.array([])))

    def test_get_median_distance_with_1_element(self):
        assert np.isnan(processing.get_median_distance(np.array([1])))

    def test_get_median_distance_with_2_elements(self):
        assert processing.get_median_distance(np.array([1, 2])) == pytest.approx(1.0)

    def test_get_median_distance_with_random_array(self):
        assert processing.get_median_distance(
            np.array([1, 3, 5, 6, 9, 16, 19, 22, 27, 30, 38, 39, 70])) == pytest.approx(3.0)

    @pytest.mark.skip(reason="fails due to lack of check for string input")
    def test_get_median_distance_with_1_string_element(
            self):  # currently returns nan, might be better to return TypeError
        with pytest.raises(TypeError):
            processing.get_median_distance(np.array(['a']))
        # assert np.isnan(processing.get_median_distance(np.array(['a'])))


class TestRunningMean:

    def test_running_mean_with_empty_array(self):
        assert np.array_equal(processing.running_mean(x=np.array([]), n=1), np.array([]))

    def test_running_mean_with_random_array(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=1),
                                   np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_random_list(self):
        np.testing.assert_allclose(processing.running_mean(x=[1, 2, 3, 4, 5], n=1), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_random_tuple(self):
        np.testing.assert_allclose(processing.running_mean(x=(1, 2, 3, 4, 5), n=1), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_n_smaller_than_array_length(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=2),
                                   np.array([1.5, 2.5, 3.5, 4.5]))
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=3), np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=4), np.array([2.5, 3.5]))

    def test_running_mean_with_n_equal_to_array_length(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=5), np.array([3.0]))

    @pytest.mark.skip(reason="fails due to lack of check for value of 'n'")
    def test_running_mean_with_invalid_n(self):
        # assert np.array_equal(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=6), np.array([]))
        with pytest.raises(ValueError):  # currently returns np.array([])
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=6)

        with pytest.raises(ValueError):
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=0)

        with pytest.raises(ValueError):  # currently returns np.array([-15.])
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=-1)

    @pytest.mark.skip(reason="fails due to lack of check for dim")
    def test_running_mean_with_2d_array(self):
        with pytest.raises(ValueError):  # currently returns np.array([1., 2., 3., 4., 5., 1., 2., 3., 4., 5.])
            processing.running_mean(x=np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), n=1)


class TestTrimEqualLen:

    def test_trim_equal_len_with_empty_list(self):
        raw = []
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

    def test_trim_equal_len_with_list_of_at_least_1_empty_arrays(self):
        raw = [np.array([]), np.array([])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

        raw = [np.array([]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), [np.array([]), np.array([])])])

    def test_trim_equal_len_with_list_of_equal_length_arrays(self):
        raw = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

    def test_trim_equal_len_with_list_of_unequal_length_arrays_in_any_order(self):
        raw = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])]
        assert all(
            [np.array_equal(a, b) for a, b in
             zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])])])

        raw = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4])]
        assert all(
            [np.array_equal(a, b) for a, b in
             zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])])])

    @pytest.mark.skip(reason="fails due to lack of check for dim of list elements")
    def test_trim_equal_len_with_list_with_at_least_1_2d_array(self):
        raw = [np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in
                    zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])])])
        # should shape of individual arrays be checked before trimming? this could add computational time if array too long.
        # could be unnecessary if input usually is list of arrays of same dimension.
        # OR: using 1 for loop instead of list comprehension to trim arrays as well as check dimensions simultaneously


class TestParseRaw:

    @pytest.fixture
    def input_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.trim_equal_len', side_effect=self.trim_equal_len)
        raw = np.array(
            [0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4, 2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        stimulation_idxs = np.array([3, 10, 14, 16, 19])
        samples_before_stim = 0
        skip_one = False
        min_len_chunk = 1
        yield raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk

    def trim_equal_len(self, raw):
        key = get_key(str(raw))
        return_values = {
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2]),
                         np.array([0.7, 0.3, 0.2]),
                         np.array([1.7])])): [np.array([3.1]), np.array([2.2]), np.array([2.6]), np.array([0.7]),
                                              np.array([1.7])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2]),
                         np.array([0.7, 0.3, 0.2])])): [np.array([3.1, 4.6]), np.array([2.2, 3.6]),
                                                        np.array([2.6, 1.2]), np.array([0.7, 0.3])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4, 2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2, 0.7, 0.3, 0.2]),
                         np.array([1.7])])): [np.array([3.1]), np.array([2.6]), np.array([1.7])],
            get_key(str([np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3]),
                         np.array([2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6]),
                         np.array([4.4, 3.5, 2.6]),
                         np.array([1.2, 0.7, 0.3, 0.2, 1.7])])): [np.array([0.4, 1.2]), np.array([2.7, 1.8]),
                                                                  np.array([2.2, 3.6]), np.array([4.4, 3.5]),
                                                                  np.array([1.2, 0.7])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])])): [np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])],
            get_key(str([])): []}

        return return_values.get(key, "default_output")

    def test_parse_raw_with_min_len_chunk_lower_than_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data

        np.testing.assert_allclose(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

        min_len_chunk = 2
        np.testing.assert_allclose(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[3.1, 4.6], [2.2, 3.6], [2.6, 1.2], [0.7, 0.3]]))

    def test_parse_raw_with_min_len_chunk_equals_length_of_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        min_len_chunk = 7

        np.testing.assert_allclose(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk), [[3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]])

    def test_parse_raw_with_min_len_chunk_greater_than_length_of_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        min_len_chunk = 8

        with pytest.raises(ValueError):  # should this return an empty list instead?
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_skip_one_equals_true(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        skip_one = True

        assert np.array_equal(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[3.1], [2.6], [1.7]]))

    @pytest.mark.skip(reason="fails due to lack of support for stimulation_idxs of type 'list'")
    def test_parse_raw_with_stimulation_idxs_as_list(self,
                                                     input_data):  # could add support for this data type since raw can work as a list
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        stimulation_idxs = [3, 10, 14, 16, 19]

        # with pytest.raises(TypeError):
        #    processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)
        np.testing.assert_allclose(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

    @pytest.mark.skip(reason="fails due to lack of constraint on length of stimulation_idxs")
    def test_parse_raw_with_empty_stimulation_idxs(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        stimulation_idxs = np.array([])

        with pytest.raises(ValueError):
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_empty_array(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        raw = np.array([])

        with pytest.raises(ValueError):  # message not very informative
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_samples_before_stim_greater_than_first_stim_id(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        samples_before_stim = 4

        np.testing.assert_allclose(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]]))


class TestGetAverageAmplitudes:

    @pytest.fixture
    def input_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.average_block', side_effect=self.average_block)
        parsed_raw = np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]])
        tested_amplitudes = [1]
        yield parsed_raw, tested_amplitudes

    def average_block(self, array: np.typing.ArrayLike, start: int, stop: int) -> np.ndarray:
        block = array[start:stop]
        key = get_key(str(block))

        return_values = {get_key(str(np.array([[0.4, 1.2],
                                               [2.7, 1.8],
                                               [2.2, 3.6],
                                               [4.4, 3.5],
                                               [1.2, 0.7]]))): [2.18, 2.16],
                         get_key(str(np.array([[0.4, 1.2],
                                               [2.7, 1.8]]))): [1.55, 1.5],
                         get_key(str(np.array([[2.2, 3.6],
                                               [4.4, 3.5]]))): [3.3, 3.55],
                         get_key(str(np.array([[0.4, 1.2]]))): [0.4, 1.2],
                         get_key(str(np.array([[2.7, 1.8]]))): [2.7, 1.8]
                         }

        return return_values.get(key, "default_output")

    def test_get_average_amplitudes_with_1_tested_amplitude(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes), [[2.18, 2.16]])

    def test_get_average_amplitudes_with_multiple_tested_amplitudes_and_default_pulses_number(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        tested_amplitudes = [1, 2]

        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes),
                                   [np.array([1.55, 1.5]), np.array([3.3, 3.55])])

    def test_get_average_amplitudes_with_multiple_tested_amplitudes_and_custom_pulses_number(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        tested_amplitudes = [1, 2]
        pulses_number = 1

        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes, pulses_number),
                                   [np.array([0.4, 1.2]), np.array([2.7, 1.8])])


class TestAverageBlock:

    @pytest.fixture
    def input_data(self):
        array = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        start = 0
        stop = 8
        yield array, start, stop

    def test_average_block_with_random_1d_input(self, input_data):
        array, start, stop = input_data

        assert processing.average_block(array, start, stop) == pytest.approx(3.5)

    def test_average_block_with_random_2d_input(self, input_data):
        array, start, stop = input_data
        array = np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]])

        np.testing.assert_allclose(processing.average_block(array, start, stop), np.array([2.18, 2.16]))

    def test_average_block_with_stop_greater_than_array_length(self, input_data):
        array, start, stop = input_data
        stop = 9

        assert processing.average_block(array, start, stop) == pytest.approx(3.5)

    def test_average_block_with_start_greater_than_array_length(self, input_data):
        array, start, stop = input_data
        start = 9

        assert np.isnan(processing.average_block(array, start, stop))  # could be better to return IndexError
        # with pytest.raises(IndexError):
        #    processing.average_block(array, start, stop)

    def test_average_block_with_empty_array(self, input_data):
        array, start, stop = input_data
        array = np.array([])

        assert np.isnan(processing.average_block(array, start, stop))  # could be better to return ValueError
        # with pytest.raises(ValueError):
        #    processing.average_block(array, start, stop)


class TestFindClosestIndex:

    def test_find_closest_index_with_empty_input(self):
        data = np.array([])
        with pytest.raises(ValueError):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_index_with_nan_input(self):
        data = np.array([np.nan, 1])
        with pytest.raises(ValueError,
                           match=r"The input array contains nan which will always return as the nearest to datapoint"):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_index_with_input_having_single_match(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        assert processing.find_closest_index(data, datapoint=3) == 3

    def test_find_closest_index_with_input_having_two_matches(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        assert processing.find_closest_index(data, datapoint=3) == 3

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])
        assert processing.find_closest_index(data, datapoint=3) == (0, 3)  # currently returns 3


class TestFindClosestSmallerIndex:

    @pytest.mark.skip(reason="fails due to lack of check for empty input")
    def test_find_closest_smaller_index_with_empty_input(self):
        data = np.array([])

        # assert processing.find_closest_smaller_index(data, datapoint=3) == 0
        with pytest.raises(ValueError):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_smaller_index_with_nan_input(self):  # should this case be handled differently?
        data = np.array([np.nan, 1])

        assert processing.find_closest_smaller_index(data, datapoint=3) == 1

    def test_find_closest_smaller_index_with_input_having_single_match(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])

        assert processing.find_closest_smaller_index(data, datapoint=3) == 6

    def test_find_closest_smaller_index_with_input_having_two_matches(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        print(processing.find_closest_smaller_index(data, datapoint=3))
        assert processing.find_closest_smaller_index(data, datapoint=3) == 5

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])

        assert processing.find_closest_index(data, datapoint=3) == (0, 5)  # currently returns 5


class TestGetSpectrogramData:

    @pytest.fixture
    def input_data(self):
        path = '../../test_data/TDT_test_data'

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr2', sync_present=True, t2=28.79)

        fs = baseline.fs
        raw = baseline.raw
        yield fs, raw

    def test_get_spectrogram_data(self, input_data):  # result too big to hardcode so checked with the first segment only
        fs, raw = input_data
        sxx_actual, _, _ = processing.get_spectrogram_data(fs, raw)
        sxx_expected = np.array([[3.92464029e-13, 2.52512511e-10, 3.88925871e-11, 1.09310330e-11,
                                  2.50914242e-12, 4.74824477e-12, 1.13642808e-12, 1.62361411e-12,
                                  9.45552772e-13, 2.22946986e-12, 2.60588551e-14, 8.55714805e-13,
                                  8.85566035e-13, 1.42646008e-12, 1.45257721e-13, 9.03181338e-14,
                                  2.68043725e-13, 1.52894394e-13, 3.09884158e-14, 7.91936910e-14,
                                  3.96048279e-14, 1.03952288e-13, 2.63260307e-13, 1.50845116e-13,
                                  3.44580809e-13, 1.09821494e-13, 1.04074789e-13, 9.53600602e-14,
                                  8.56902440e-14, 3.88436198e-14, 1.63998894e-14, 4.25312819e-15,
                                  2.50111536e-13, 9.61959191e-14, 2.05672786e-13, 5.77824396e-14,
                                  5.24750728e-16, 3.58739290e-15, 1.74995753e-14, 1.11471047e-13,
                                  2.19020291e-14, 3.62651024e-14, 2.56354335e-14, 8.64429920e-14,
                                  3.00472436e-14, 1.07050091e-13, 2.93383499e-15, 1.55127322e-13,
                                  5.04462296e-14, 3.53996854e-14, 4.01521366e-14, 1.17832236e-14,
                                  5.49950699e-16, 1.35346121e-13, 5.53881248e-14, 6.02071358e-15,
                                  1.46835045e-13, 2.26777783e-15, 5.31139463e-14, 2.21710095e-14,
                                  6.40940853e-15, 4.51449249e-14, 1.00093282e-14, 3.83340515e-14,
                                  2.66548817e-15, 2.72232899e-14, 1.58304576e-13, 3.85474191e-15,
                                  2.81871457e-14, 2.85015575e-14, 4.03910710e-14, 4.62958986e-16,
                                  4.49077217e-14, 1.84744611e-14, 1.57298694e-14, 2.90248951e-14,
                                  2.01362347e-14, 9.67128413e-15, 1.21049182e-14, 1.61885293e-14,
                                  2.54324116e-14, 1.52832323e-14, 7.79378393e-15, 1.63342054e-14,
                                  6.57018173e-15, 2.30945634e-14, 1.96377897e-15, 1.85628575e-14,
                                  6.78447844e-14, 1.37715455e-15, 1.30947380e-14, 7.62977463e-15,
                                  1.78785446e-14, 2.19068456e-13, 2.33242194e-14, 5.55866896e-14,
                                  8.19471173e-15, 2.93841743e-15, 1.84445507e-14, 4.15043857e-15,
                                  9.98755776e-15, 1.24475356e-16, 6.35753115e-15, 2.35987953e-14,
                                  1.34041007e-14, 2.37877175e-14, 1.02335751e-14, 3.91081295e-15,
                                  6.74944855e-14, 2.52388307e-14, 1.73601418e-14, 1.69430846e-15,
                                  3.21049543e-14, 8.98547797e-15, 4.89796666e-14, 3.71928906e-15,
                                  1.71326407e-14, 6.77017375e-14, 1.32239639e-14, 2.16658640e-15,
                                  4.93751556e-15, 2.20274878e-15, 2.15663448e-14, 4.28068801e-14,
                                  5.25195113e-15, 1.50496477e-14, 1.47414141e-15, 1.29712889e-14,
                                  6.97019304e-15],
                                 [3.84131615e-12, 1.49168705e-10, 4.80426358e-11, 5.42207945e-12,
                                  6.40392600e-12, 1.32967509e-11, 1.77156196e-12, 3.36015157e-12,
                                  2.37189760e-12, 4.64576619e-13, 3.49476741e-13, 2.46165257e-13,
                                  8.17362508e-13, 2.11964608e-14, 3.84361136e-13, 1.96616215e-13,
                                  2.13639599e-13, 9.28242129e-14, 2.11862957e-13, 1.77119224e-13,
                                  2.38331490e-13, 4.95655898e-14, 1.72894685e-13, 5.55010844e-13,
                                  4.17290036e-14, 2.50505698e-13, 1.04380412e-14, 2.18613376e-14,
                                  1.11927620e-14, 7.12224808e-14, 5.32446537e-14, 2.57744591e-13,
                                  3.48819450e-14, 1.06708304e-14, 4.85756613e-16, 1.90426745e-14,
                                  1.33842243e-14, 7.32449312e-14, 2.33667198e-13, 8.22647817e-14,
                                  6.11408642e-14, 5.54676815e-14, 6.31055809e-14, 3.69915449e-14,
                                  5.32274996e-14, 3.61986984e-14, 9.40404354e-15, 2.62621617e-14,
                                  4.87324618e-14, 2.47673603e-14, 9.49282970e-14, 8.02068237e-14,
                                  8.85751501e-14, 1.71849527e-13, 8.16654123e-12, 5.13399653e-12,
                                  5.97955821e-13, 2.10874368e-13, 6.34364726e-14, 4.61623097e-14,
                                  1.56609023e-14, 5.15065489e-14, 7.64357653e-14, 7.99846707e-14,
                                  3.18960082e-14, 6.74555779e-15, 1.71087374e-14, 6.34879925e-14,
                                  2.61783291e-14, 1.19931531e-13, 6.99621405e-13, 2.82928932e-12,
                                  1.08432692e-13, 1.25113610e-14, 3.45399368e-14, 6.29784717e-14,
                                  9.09531239e-14, 1.46890353e-13, 3.22502577e-14, 1.03335817e-14,
                                  1.61548580e-14, 1.17735462e-14, 8.09630344e-14, 3.01512558e-14,
                                  1.21575479e-13, 1.03677500e-15, 9.91934738e-14, 1.28879764e-14,
                                  1.96735548e-14, 4.83660683e-16, 1.15181344e-14, 2.50454801e-14,
                                  1.91438011e-13, 4.68417297e-13, 7.62259789e-14, 7.99558762e-16,
                                  1.32194907e-14, 4.29460612e-14, 1.13700451e-14, 4.45617808e-15,
                                  1.59662102e-14, 1.95884348e-14, 1.31105996e-14, 2.08340438e-15,
                                  2.18210866e-14, 7.12172766e-15, 6.66604384e-15, 3.38586953e-14,
                                  5.71702923e-14, 2.97900376e-13, 9.87003363e-15, 5.02398069e-15,
                                  1.77557401e-14, 1.29710451e-13, 1.72533107e-14, 1.94925257e-15,
                                  4.06687216e-14, 3.60827353e-15, 2.73787273e-14, 2.18440090e-14,
                                  4.67823073e-14, 5.91957920e-14, 1.68511699e-14, 1.16006710e-14,
                                  4.78767798e-15, 3.00687739e-15, 3.99398777e-15, 1.66929689e-13,
                                  2.44761309e-15],
                                 [3.43867708e-12, 1.37757764e-10, 1.60199163e-11, 2.86099421e-12,
                                  1.58849203e-12, 2.74494967e-12, 3.07869745e-12, 1.39741257e-12,
                                  1.23017525e-12, 9.03048849e-13, 7.59272948e-14, 1.46184077e-13,
                                  8.77268635e-13, 1.36064472e-13, 2.37785025e-13, 9.14765835e-14,
                                  1.86142470e-14, 1.93899977e-13, 1.64055184e-13, 1.21849742e-13,
                                  2.31226035e-13, 1.77724622e-14, 1.10302579e-14, 2.70744527e-13,
                                  8.62650473e-14, 1.11816765e-13, 8.11077923e-15, 2.50056895e-15,
                                  1.07794219e-13, 8.21061088e-14, 6.24650207e-14, 4.05638576e-13,
                                  1.59618192e-14, 1.92484188e-14, 1.70010422e-14, 8.45619215e-14,
                                  2.43498713e-14, 3.77081721e-14, 2.73985153e-13, 3.42131960e-15,
                                  3.01318114e-14, 4.52225004e-15, 2.87510596e-14, 1.67117107e-13,
                                  1.91276879e-14, 1.71627401e-13, 5.83068750e-14, 1.36409760e-14,
                                  5.27358817e-14, 4.54175475e-14, 6.01512587e-14, 1.88160292e-13,
                                  2.07419666e-14, 4.64492973e-13, 8.60022557e-12, 6.49034776e-12,
                                  1.17559391e-12, 9.19470527e-14, 1.43847499e-13, 4.81226014e-14,
                                  3.37247252e-14, 1.29608984e-13, 1.66534195e-15, 4.62950059e-14,
                                  3.01449302e-14, 1.20812640e-14, 7.58583395e-14, 1.43596859e-13,
                                  4.04712918e-14, 8.35576589e-14, 1.97110327e-13, 1.27413911e-12,
                                  9.31100222e-14, 1.34298863e-13, 4.28945853e-14, 9.03741125e-14,
                                  3.24291104e-14, 7.87295305e-14, 2.01388046e-14, 1.62898029e-13,
                                  6.46472757e-14, 5.90061379e-14, 1.05771051e-13, 1.12577130e-13,
                                  7.56720397e-14, 6.98121303e-14, 1.28041620e-13, 3.76505264e-14,
                                  1.27822327e-13, 9.50425041e-14, 3.83885767e-14, 4.10709369e-14,
                                  1.19788456e-13, 6.55877696e-13, 5.37983388e-14, 1.02047345e-13,
                                  5.79841215e-14, 4.62214698e-15, 3.19044277e-14, 1.12038410e-13,
                                  8.11436726e-15, 1.05721326e-13, 1.30796090e-13, 5.96648450e-14,
                                  5.00762389e-14, 2.41806748e-14, 6.99866224e-15, 3.46930262e-14,
                                  7.39452309e-14, 2.30204392e-13, 2.43022426e-14, 1.91050734e-13,
                                  4.83196010e-14, 1.57194367e-13, 1.78146496e-14, 3.52239181e-15,
                                  4.73548745e-14, 1.30862313e-15, 2.33330777e-14, 9.58772856e-14,
                                  1.14226899e-14, 4.20637790e-15, 3.00350870e-14, 2.81351853e-14,
                                  1.32066582e-14, 5.50038835e-14, 4.69276480e-14, 9.34772753e-14,
                                  3.11434406e-15],
                                 [2.92567359e-12, 2.56295152e-10, 2.22751990e-11, 5.13365080e-11,
                                  3.17184668e-12, 7.87968128e-12, 8.62455249e-14, 1.16909878e-12,
                                  1.91491178e-13, 3.56238422e-13, 2.26542336e-13, 1.58577578e-13,
                                  6.59516793e-14, 3.00289931e-13, 4.34980692e-14, 3.93576129e-14,
                                  3.66594369e-14, 2.19367737e-13, 9.00671952e-14, 2.27459653e-13,
                                  2.82353784e-13, 7.79103209e-14, 1.02662426e-13, 3.10832360e-13,
                                  5.41424519e-15, 9.08781819e-15, 5.55760509e-14, 1.91811289e-14,
                                  2.34777374e-14, 3.94348928e-14, 1.60371282e-13, 2.69587141e-13,
                                  1.28224686e-14, 2.88423951e-15, 1.25538770e-14, 1.03033799e-14,
                                  3.74192356e-14, 1.41075603e-15, 2.19137900e-13, 3.50173479e-15,
                                  1.07819434e-13, 7.03968066e-14, 3.20555621e-14, 1.10192705e-13,
                                  1.87073105e-14, 1.83659223e-14, 3.60281669e-14, 1.17729543e-13,
                                  7.82397218e-15, 1.06343231e-13, 5.28496585e-14, 4.52060230e-14,
                                  9.40720551e-15, 7.89092966e-13, 1.47584323e-11, 1.01136721e-11,
                                  1.28086073e-12, 1.69293588e-13, 3.27906063e-15, 2.58644068e-14,
                                  8.58356423e-14, 2.19815323e-13, 1.36637917e-14, 1.98676457e-13,
                                  3.37877512e-14, 8.07295836e-15, 2.79765021e-14, 8.92133928e-14,
                                  9.09342182e-14, 1.36020210e-13, 1.52426466e-12, 5.38138978e-12,
                                  1.62603140e-13, 1.95329457e-14, 6.80848742e-14, 2.48051037e-13,
                                  6.06705780e-14, 1.20341671e-13, 3.01678950e-14, 9.87965084e-15,
                                  6.94840032e-14, 2.92339395e-14, 1.02464241e-13, 9.00929061e-15,
                                  7.53951412e-14, 4.73050410e-15, 7.71516978e-14, 2.39927059e-15,
                                  2.33323577e-14, 9.55950119e-15, 3.20698405e-15, 8.29449644e-15,
                                  2.68108967e-14, 6.69450389e-14, 1.92455440e-14, 9.45224141e-15,
                                  5.49034686e-15, 5.33640820e-14, 3.18278051e-14, 1.46180713e-15,
                                  9.03250439e-15, 2.71548598e-14, 1.41207406e-14, 6.05020070e-16,
                                  1.17310919e-15, 1.24189735e-14, 5.34283607e-15, 3.26620953e-14,
                                  6.93374327e-14, 1.69768808e-13, 1.45516940e-14, 1.41770632e-14,
                                  9.54541334e-16, 7.84618274e-14, 1.66330979e-14, 7.71447830e-17,
                                  3.43724852e-14, 4.22623410e-16, 2.40219865e-14, 1.36018643e-14,
                                  5.69503415e-14, 1.04340114e-13, 7.70777366e-15, 6.54842120e-14,
                                  9.19217519e-15, 5.76679275e-14, 2.35296029e-14, 1.11830704e-13,
                                  1.26732172e-14]])

        np.testing.assert_allclose(sxx_actual[:, :, 0]*1e15, sxx_expected*1e15)


class TestCalculatePowerSpectralDensity:

    @pytest.fixture
    def input_data(self):
        path = '../../test_data/TDT_test_data'

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr2', sync_present=True, t2=28.79)

        fs = baseline.fs
        raw = baseline.raw
        yield fs, raw

    def test_calculate_power_spectral_density(self, input_data):
        fs, data = input_data
        _, pxx_actual = processing.calculate_power_spectral_density(data, fs)
        np.set_printoptions(threshold=np.inf)
        pxx_expected = np.array([[2.08923365e-11, 1.22318725e-10, 3.55222761e-11, 1.15818223e-11,
                                  6.37188999e-12, 4.46789871e-12, 3.33765199e-12, 2.08079214e-12,
                                  1.55511715e-12, 1.20283666e-12, 9.39583589e-13, 8.74927951e-13,
                                  7.03365426e-13, 6.14591386e-13, 6.07692987e-13, 5.29488400e-13,
                                  5.43920486e-13, 5.89542575e-13, 4.26581486e-13, 3.39543172e-13,
                                  3.55481025e-13, 3.89222969e-13, 2.78528231e-13, 2.81010078e-13,
                                  3.39866021e-13, 2.55766870e-13, 2.07278652e-13, 2.73064557e-13,
                                  2.89835403e-13, 1.85189578e-13, 2.25192044e-13, 2.94058967e-13,
                                  1.84238150e-13, 2.13337459e-13, 3.17834246e-13, 2.16800482e-13,
                                  1.42874753e-13, 2.36416193e-13, 2.64889537e-13, 1.29914634e-13,
                                  1.93910982e-13, 2.99662313e-13, 1.59789757e-13, 1.53631800e-13,
                                  2.92410573e-13, 2.19465708e-13, 1.13834289e-13, 2.49500724e-13,
                                  2.87045290e-13, 1.11724703e-13, 1.94029458e-13, 3.33128219e-13,
                                  1.62007343e-13, 1.32405683e-13, 3.02668725e-13, 2.36358432e-13,
                                  1.01286465e-13, 2.49586078e-13, 3.19007407e-13, 1.15094593e-13,
                                  1.81793206e-13, 3.47212276e-13, 1.74723436e-13, 1.23835729e-13,
                                  3.40520364e-13, 2.88252034e-13, 9.67781085e-14, 2.76070264e-13,
                                  3.63380170e-13, 1.25249935e-13, 1.81754609e-13, 3.87151059e-13,
                                  2.07294034e-13, 1.12448537e-13, 3.45550194e-13, 3.12431613e-13,
                                  9.07428633e-14, 2.67559981e-13, 3.95273766e-13, 1.31831151e-13,
                                  1.72685190e-13, 4.07960287e-13, 2.32868927e-13, 1.08981394e-13,
                                  3.60621824e-13, 3.45942404e-13, 9.73498016e-14, 2.69265458e-13,
                                  4.22495535e-13, 1.56291714e-13, 1.76531600e-13, 4.34664539e-13,
                                  2.71063391e-13, 1.35960701e-13, 5.00791791e-13, 4.74430879e-13,
                                  1.08078761e-13, 2.70533948e-13, 4.46147405e-13, 1.72995678e-13,
                                  1.66806212e-13, 4.40027328e-13, 2.89291594e-13, 1.03768597e-13,
                                  3.72296432e-13, 4.00810217e-13, 1.11161162e-13, 2.52959139e-13,
                                  4.54856476e-13, 1.94830317e-13, 1.59300877e-13, 4.50657876e-13,
                                  3.12134108e-13, 9.96279272e-14, 3.63284110e-13, 4.20370824e-13,
                                  1.23133817e-13, 2.53259165e-13, 4.75908863e-13, 2.11305542e-13,
                                  1.44565661e-13, 4.47557681e-13, 3.43306194e-13, 1.00186142e-13,
                                  3.47995829e-13, 4.38355569e-13, 1.38481417e-13, 2.35738322e-13,
                                  2.39142094e-13],
                                 [4.52297332e-11, 3.29342525e-10, 1.28062019e-10, 6.36211223e-11,
                                  4.16166338e-11, 3.42750481e-11, 2.91315375e-11, 2.41740013e-11,
                                  2.11575670e-11, 1.94240665e-11, 1.85395622e-11, 1.75181952e-11,
                                  1.64988318e-11, 1.55240699e-11, 1.48074053e-11, 1.41097576e-11,
                                  1.29966515e-11, 1.18818488e-11, 1.11004886e-11, 1.02824017e-11,
                                  9.50947220e-12, 8.90213337e-12, 8.08781341e-12, 7.36317517e-12,
                                  6.79728582e-12, 6.03066165e-12, 5.30871137e-12, 4.75616768e-12,
                                  4.26629738e-12, 3.68459733e-12, 3.18223833e-12, 2.75966554e-12,
                                  2.26755398e-12, 1.83628807e-12, 1.60965382e-12, 1.28034226e-12,
                                  9.44045406e-13, 8.21741709e-13, 7.63307494e-13, 5.02795288e-13,
                                  4.38413628e-13, 5.00191739e-13, 3.24765497e-13, 2.56295202e-13,
                                  3.40289591e-13, 3.83089502e-13, 3.60777353e-13, 4.38933856e-13,
                                  5.83777330e-13, 5.36733255e-13, 6.50565105e-13, 8.72161555e-13,
                                  8.61016282e-13, 4.24472870e-12, 1.00958001e-11, 2.82068965e-12,
                                  1.08576071e-12, 1.21233297e-12, 1.37487276e-12, 1.24386124e-12,
                                  1.26250323e-12, 1.43842859e-12, 1.37696354e-12, 1.33296857e-12,
                                  1.46940695e-12, 1.54780409e-12, 1.94257501e-12, 2.98397570e-12,
                                  2.29160723e-12, 1.34377589e-12, 1.14347019e-12, 1.25641717e-12,
                                  1.00286001e-12, 7.43299573e-13, 7.91853670e-13, 7.49084822e-13,
                                  5.06786236e-13, 5.12026023e-13, 5.74087761e-13, 3.68220455e-13,
                                  3.15198415e-13, 4.61109016e-13, 3.43138441e-13, 1.74103285e-13,
                                  3.25514653e-13, 3.37147492e-13, 1.36620384e-13, 2.09623294e-13,
                                  3.79280862e-13, 2.08648975e-13, 1.96463830e-13, 4.10388357e-13,
                                  3.93705576e-13, 3.57444136e-13, 8.81283219e-13, 8.99533757e-13,
                                  4.53203040e-13, 5.41535078e-13, 7.06372189e-13, 5.70243830e-13,
                                  5.82801711e-13, 7.93196400e-13, 7.22853472e-13, 6.11244941e-13,
                                  8.09546928e-13, 9.06601888e-13, 6.91932948e-13, 8.15828687e-13,
                                  9.85905801e-13, 7.17283004e-13, 6.51442984e-13, 8.25877344e-13,
                                  7.62772712e-13, 5.87555287e-13, 6.64219222e-13, 6.67884639e-13,
                                  4.28945074e-13, 4.72020535e-13, 5.91541139e-13, 3.78901825e-13,
                                  2.91229199e-13, 4.77617999e-13, 4.10357593e-13, 1.82772390e-13,
                                  3.14116219e-13, 3.57284677e-13, 1.60326220e-13, 2.16083987e-13,
                                  1.79512506e-13],
                                 [3.96716097e-11, 2.71313472e-10, 9.30817506e-11, 4.11308973e-11,
                                  2.55586090e-11, 2.35427268e-11, 2.11994207e-11, 1.79254458e-11,
                                  1.59395865e-11, 1.44614572e-11, 1.34770832e-11, 1.29392720e-11,
                                  1.22208511e-11, 1.15500482e-11, 1.10975014e-11, 1.03727166e-11,
                                  9.56704854e-12, 8.84039803e-12, 8.31357812e-12, 7.82421176e-12,
                                  7.31596814e-12, 6.88493403e-12, 6.17053717e-12, 5.64707482e-12,
                                  5.30559190e-12, 4.68254385e-12, 4.09920318e-12, 3.72573717e-12,
                                  3.46426121e-12, 3.00671705e-12, 2.57997356e-12, 2.29071796e-12,
                                  1.91896585e-12, 1.57848593e-12, 1.45019847e-12, 1.18604790e-12,
                                  8.94845775e-13, 8.26989302e-13, 8.17970691e-13, 5.69692080e-13,
                                  5.09581093e-13, 5.82479052e-13, 4.26545572e-13, 3.40612819e-13,
                                  4.20583409e-13, 4.45491571e-13, 3.59558330e-13, 4.20596555e-13,
                                  5.61403192e-13, 4.60682816e-13, 5.34443854e-13, 7.27889916e-13,
                                  6.68186264e-13, 4.46803141e-12, 1.12673352e-11, 3.04962869e-12,
                                  8.41112444e-13, 9.04185797e-13, 1.07256391e-12, 8.88013838e-13,
                                  8.88872851e-13, 1.05862899e-12, 9.91819040e-13, 9.56390240e-13,
                                  1.10378547e-12, 1.17442351e-12, 1.34201168e-12, 1.98242204e-12,
                                  1.61129508e-12, 9.52722384e-13, 8.26068760e-13, 9.94142377e-13,
                                  7.98876698e-13, 5.91165571e-13, 7.08270411e-13, 7.02104390e-13,
                                  4.43139638e-13, 4.88731072e-13, 5.83592148e-13, 3.63222419e-13,
                                  3.34727634e-13, 5.44349505e-13, 4.13985930e-13, 2.25757496e-13,
                                  4.17862983e-13, 4.24790628e-13, 1.96218299e-13, 2.95885576e-13,
                                  4.69900919e-13, 2.45280115e-13, 2.34716326e-13, 4.70080680e-13,
                                  4.12755198e-13, 3.23213163e-13, 6.61141443e-13, 6.83776224e-13,
                                  3.67828110e-13, 4.97189909e-13, 6.81304459e-13, 4.60843440e-13,
                                  4.61774987e-13, 7.25121677e-13, 6.07177720e-13, 4.43962955e-13,
                                  6.88940658e-13, 7.86727454e-13, 5.21147687e-13, 6.44583074e-13,
                                  8.35126673e-13, 5.91612588e-13, 5.21012432e-13, 7.37768462e-13,
                                  6.74393646e-13, 4.65720344e-13, 6.03490727e-13, 6.38105942e-13,
                                  3.63170269e-13, 4.61385921e-13, 6.14581519e-13, 3.66954107e-13,
                                  2.84270708e-13, 5.32094930e-13, 4.61423488e-13, 2.05779621e-13,
                                  4.02731017e-13, 4.66364631e-13, 2.17850518e-13, 2.94322102e-13,
                                  2.41492698e-13],
                                 [1.23815038e-11, 8.83132525e-11, 4.03585151e-11, 3.64109992e-11,
                                  2.46222192e-11, 1.65851465e-11, 1.52342843e-11, 1.28334236e-11,
                                  1.13762003e-11, 1.02976187e-11, 9.35514600e-12, 8.83557203e-12,
                                  8.56626142e-12, 8.32681060e-12, 8.08215821e-12, 7.52225279e-12,
                                  7.19982060e-12, 6.91954090e-12, 6.53708121e-12, 6.11396871e-12,
                                  5.75561127e-12, 5.39697020e-12, 4.89671585e-12, 4.50131989e-12,
                                  4.22898392e-12, 3.77153647e-12, 3.40774154e-12, 3.12231903e-12,
                                  2.86826661e-12, 2.52174865e-12, 2.22860554e-12, 2.03249331e-12,
                                  1.69910093e-12, 1.44119059e-12, 1.31969782e-12, 1.11088818e-12,
                                  8.65615034e-13, 7.99852263e-13, 7.49508420e-13, 5.33037481e-13,
                                  5.18718261e-13, 5.51822856e-13, 3.78219347e-13, 3.03770789e-13,
                                  3.58846010e-13, 3.52086849e-13, 3.04650267e-13, 3.64005186e-13,
                                  4.44708669e-13, 3.39896107e-13, 3.90758606e-13, 5.41229876e-13,
                                  5.02700420e-13, 6.93123423e-12, 1.79328132e-11, 4.05008492e-12,
                                  6.20023238e-13, 6.53107071e-13, 7.81145927e-13, 6.50002946e-13,
                                  6.78159243e-13, 8.23850049e-13, 7.51031615e-13, 7.25271893e-13,
                                  8.90502353e-13, 1.07067426e-12, 1.89808759e-12, 3.62851199e-12,
                                  2.46355150e-12, 1.06763231e-12, 7.09132839e-13, 7.90394822e-13,
                                  6.47416962e-13, 4.87036518e-13, 5.62482244e-13, 5.45540555e-13,
                                  3.39217505e-13, 3.86994907e-13, 4.84481812e-13, 2.89905984e-13,
                                  2.59241034e-13, 4.29306357e-13, 3.09201910e-13, 1.54006446e-13,
                                  3.22050898e-13, 3.28645639e-13, 1.29495861e-13, 2.04038121e-13,
                                  3.49848948e-13, 1.47102599e-13, 1.27749062e-13, 3.22680847e-13,
                                  2.54587421e-13, 1.50254226e-13, 3.26929320e-13, 3.58900572e-13,
                                  1.77907602e-13, 2.74673350e-13, 4.14567252e-13, 2.40068192e-13,
                                  2.34034065e-13, 4.33559953e-13, 3.40881945e-13, 2.08779310e-13,
                                  4.07474103e-13, 4.96017073e-13, 2.70556310e-13, 4.09077476e-13,
                                  6.03563857e-13, 3.33099786e-13, 2.70280299e-13, 4.71464935e-13,
                                  4.26954533e-13, 2.59050837e-13, 3.77161599e-13, 4.13630068e-13,
                                  1.92280938e-13, 2.69704316e-13, 4.22959438e-13, 2.27018626e-13,
                                  1.65303426e-13, 3.87667031e-13, 3.30109692e-13, 1.17251003e-13,
                                  2.76254985e-13, 3.41873448e-13, 1.37757197e-13, 2.08928632e-13,
                                  1.86319073e-13]])
        np.testing.assert_allclose(pxx_actual*1e15, pxx_expected*1e15)
