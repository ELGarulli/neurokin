import numpy as np
from neurokin.utils.neural import processing, importing
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

    def test_simply_mean_data_binarize_with_nan_array(self):
        with pytest.raises(ValueError):
            processing.simply_mean_data_binarize(np.array([1, np.nan]))

    def test_simply_mean_data_binarize_with_2d_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([[1, 1, 2], [1, 2, 3]])),
                              np.array([[0, 0, 1], [0, 1, 1]]))


class TestGetStimTimestamps:

    def test_get_stim_timestamps_with_empty_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([])), np.array([]))

    def test_get_stim_timestamps_with_nan_array(self):
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([np.nan]))

    def test_get_stim_timestamps_with_2d_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([[1, 2, 3], [0, 2, 3]])), np.array([[0], [1]]))

    def test_get_stim_timestamps_with_random_array_without_expected_pulses(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_greater_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=3),
            np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_exact_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=2),
            np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_lower_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=1),
            np.array([1]))

    def test_get_stim_timestamps_with_random_array_with_0_expected_pulses(self):
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=0)

    def test_get_stim_timestamps_with_random_array_with_negative_expected_pulses(self):
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-1)

    def test_get_stim_timestamps_with_random_array_mid_pulse(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([1, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([0, 8]))

    def test_get_stim_timestamps_with_0_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 0, 0, 0, 0, 0])), np.array([]))


class TestGetTTimestampsStimBlocks:

    @pytest.fixture
    def neural_data(self, monkeypatch):
        monkeypatch.setattr(processing, 'get_stim_timestamps', self.get_stim_timestamps)
        monkeypatch.setattr(importing, 'time_to_sample', self.time_to_sample)
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

    def test_get_median_distance_with_1_string_element(
            self):
        with pytest.raises(TypeError):
            processing.get_median_distance(np.array(['a']))


class TestRunningMean:

    def test_running_mean_with_empty_array(self):
        with pytest.raises(ValueError):
            processing.running_mean(x=np.array([]), n=1)

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

    def test_running_mean_with_invalid_n(self):
        with pytest.raises(ValueError):  # currently returns np.array([])
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=6)

        with pytest.raises(ValueError):
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=0)

        with pytest.raises(ValueError):
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=-1)

    def test_running_mean_with_2d_array(self):
        with pytest.raises(ValueError):
            np.array_equal(processing.running_mean(x=np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), n=5))


class TestRunningMean2D:
    def test_running_mean_with_2d_array(self):
        assert np.array_equal(processing.running_mean_2D(x=np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), n=5),
                              np.array([[3], [3]]))

    def test_running_mean_1D_array(self):
        with pytest.raises(ValueError):
            processing.running_mean_2D(x=np.array([1, 2, 3, 4, 5]), n=5)


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

    def test_trim_equal_len_with_list_with_at_least_1_2d_array(self):
        with pytest.raises(ValueError):
            raw = [np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), np.array([1, 2, 3, 4, 5])]
            processing.trim_equal_len(raw)


class TestParseRaw:

    @pytest.fixture
    def input_data(self, monkeypatch):
        monkeypatch.setattr(processing, 'trim_equal_len', self.trim_equal_len)
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
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])])): [
                np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])],
            get_key(str([])): []}

        return return_values.get(key, "default_output")

    def test_parse_raw_with_min_len_chunk_lower_than_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

        min_len_chunk = 2
        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1, 4.6], [2.2, 3.6], [2.6, 1.2], [0.7, 0.3]]))

    def test_parse_raw_with_min_len_chunk_equals_length_of_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        min_len_chunk = 7

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            [[3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]])

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

    def test_parse_raw_with_stimulation_idxs_as_list(self,
                                                     input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        stimulation_idxs = [3, 10, 14, 16, 19]

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

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

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]]))


class TestGetAverageAmplitudes:

    @pytest.fixture
    def input_data(self, monkeypatch):
        monkeypatch.setattr(processing, 'average_block', self.average_block)
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

    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])
        assert processing.find_closest_index(data, datapoint=3) == (0, 3)  # currently returns 3


class TestFindClosestSmallerIndex:

    def test_find_closest_smaller_index_with_empty_input(self):
        data = np.array([])
        with pytest.raises(ValueError):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_smaller_index_with_nan_input(self):
        data = np.array([np.nan, 1])
        with pytest.raises(ValueError):
            processing.find_closest_smaller_index(data, datapoint=3)

    def test_find_closest_smaller_index_with_input_having_single_match(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])

        assert processing.find_closest_smaller_index(data, datapoint=3) == 6

    def test_find_closest_smaller_index_with_input_having_two_matches(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        print(processing.find_closest_smaller_index(data, datapoint=3))
        assert processing.find_closest_smaller_index(data, datapoint=3) == 5

    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])

        assert processing.find_closest_index(data, datapoint=3) == (0, 3)  # currently returns 5


class TestGetSpectrogramData:

    @pytest.fixture
    def input_data(self, repo_root):
        path = repo_root / "tests" / "test_data" / "TDT_test_data"

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr2', sync_present=True, t2=14.61)

        fs = baseline.fs
        raw = baseline.raw[0, :300]
        yield fs, raw

    def test_get_spectrogram_data(self,
                                  input_data):  # result too big to hardcode so checked with the first segment only
        fs, raw = input_data
        sxx_actual, _, _ = processing.get_spectrogram_data(fs, raw)
        sxx_expected = np.array([[2.25538468e-12],
                                 [3.70232983e-10],
                                 [7.96814871e-12],
                                 [2.08931084e-11],
                                 [6.30049918e-12]])

        np.testing.assert_allclose(sxx_actual[:5] * 1e15, sxx_expected * 1e15)


class TestCalculatePowerSpectralDensity:

    @pytest.fixture
    def input_data(self, repo_root):
        path = repo_root / "tests" / "test_data" / "TDT_test_data"

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr2', sync_present=True, t2=14.61)

        fs = baseline.fs
        raw = baseline.raw[0, :300]
        yield fs, raw

    def test_calculate_power_spectral_density(self, input_data):
        fs, data = input_data
        _, pxx_actual = processing.calculate_power_spectral_density(data, fs)
        np.set_printoptions(threshold=np.inf)
        pxx_expected = np.array([6.50348915e-12, 1.92258709e-10, 2.70964206e-11, 2.33073751e-11,
                                 1.30796632e-11])
        np.testing.assert_allclose(pxx_actual[:5] * 1e15, pxx_expected * 1e15)
