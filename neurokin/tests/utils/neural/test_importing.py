import numpy as np
import pytest
from neurokin.utils.neural import importing


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


class TestTimeToSample:

    def test_time_to_sample_normal(self):
        assert importing.time_to_sample(timestamp=1, fs=40, is_t1=False, is_t2=False) == 40
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=False, is_t2=False) == 60
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=False, is_t2=False) == 70

    def test_time_to_sample_with_is_t1_equal_true(self):
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=True, is_t2=False) == 60
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=True, is_t2=False) == 71

    def test_time_to_sample_with_is_t2_equal_true(self):
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=False, is_t2=True) == 59
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=False, is_t2=True) == 70
        assert importing.time_to_sample(timestamp=28.72, fs=24414.0625, is_t2=True) == 701171

    def test_time_to_sample_with_timestamp_equal_zero(self):
        assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=False) == 0
        assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=True, is_t2=False) == 0
        assert importing.time_to_sample(timestamp=0, fs=24414.0625, is_t1=True, is_t2=False) == 0

    @pytest.mark.skip(reason="fails due to lack of check for value of 'timestamp' parameter when 'is_t2' is 'True'")
    def test_time_to_sample_with_timestamp_equal_zero_and_is_t2_equal_true(self):
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=True)
        # assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=True) == -1  # maybe should process timestamp == 0 case differently so that output stays positive? should correct output be 0 whenever timestamp == 0 no matter t1 or t2?

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'timestamp' parameter")
    def test_time_to_sample_with_negative_timestamp(self):
        # assert importing.time_to_sample(timestamp=-1, fs=40.2, is_t1=False, is_t2=False) == -40  # this should probably throw ValueError, unless time is allowed to be in the negative direction
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=-1, fs=40.2, is_t1=False, is_t2=False)

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'fs' parameter")
    def test_time_to_sample_with_fs_equal_0():
        # assert importing.time_to_sample(timestamp=1.75, fs=0, is_t1=False, is_t2=False) == 0  # maybe should throw ValueError?
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1.75, fs=0, is_t1=False, is_t2=False)

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'fs' parameter")
    def test_time_to_sample_with_negative_fs(self):
        # assert importing.time_to_sample(timestamp=1, fs=-40.2, is_t1=False, is_t2=False) == -40  # this should probably throw ValueError
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1, fs=-40.2, is_t1=False, is_t2=False)


class TestImportTDTChannelData:

    def time_to_sample(self, timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
        key = get_key(timestamp)

        return_values = {get_key(0): 0,
                         get_key(28.72): 701171}

        return return_values.get(key, 'default_output')

    @pytest.fixture
    def folderpath(self):
        yield '../../test_data/TDT_test_data'

    @pytest.fixture
    def all_channels_output(self, folderpath, mocker):
        mocker.patch('neurokin.utils.neural.importing.time_to_sample', side_effect=self.time_to_sample)
        fs, raw, stim_data, fs_sync = importing.import_tdt_channel_data(folderpath, t2=28.72)
        yield fs, raw, stim_data, fs_sync

    def test_import_tdt_channel_data_all_channels(self, all_channels_output):
        fs, raw, stim_data, fs_sync = all_channels_output

        assert fs == pytest.approx(24414.0625) and stim_data is None and fs_sync is None
        assert np.sum(raw) == pytest.approx(427456900.0)
        assert raw[3][238923] == pytest.approx(256.0)
        np.testing.assert_allclose(raw[5][252963:253963],
                                   np.array([256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
                                             256., 256., 256., 256., 256., 256., 256., 261., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                             5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]))

    def test_import_tdt_channel_data_single_channel(self, all_channels_output, folderpath):
        fs, raw, stim_data, fs_sync = all_channels_output

        np.testing.assert_allclose(raw[2], importing.import_tdt_channel_data(folderpath, ch=3, t2=28.72)[1])
        np.testing.assert_allclose(raw[0], importing.import_tdt_channel_data(folderpath, ch=1, t2=28.72)[1])

    def test_import_tdt_channel_data_multiple_channels(self, all_channels_output, folderpath):
        fs, raw, stim_data, fs_sync = all_channels_output

        np.testing.assert_allclose(raw[2:5], importing.import_tdt_channel_data(folderpath, ch=[3, 4, 5], t2=28.72)[1])


class TestImportOpenEphysChannelData:

    @pytest.fixture
    def input_data(self):
        folderpath = '../../test_data/OE_test_data'
        experiment = 'Record Node 104/experiment1'
        recording = 'recording1'
        source_processor = 'Acquisition_Board-100.Rhythm Data'
        yield folderpath, experiment, recording, source_processor

    @pytest.fixture
    def all_channels_output(self, input_data):
        folderpath, experiment, recording, source_processor = input_data
        fs, neural_data_au, sync_data = importing.import_open_ephys_channel_data(folderpath, experiment, recording,
                                                                                 source_processor=source_processor)
        yield fs, neural_data_au, sync_data

    def test_import_open_ephys_channel_data_all_channels(self, all_channels_output):
        fs, neural_data_au, sync_data = all_channels_output
        assert fs == pytest.approx(30000.0) and sync_data is None
        assert np.sum(neural_data_au) == 9592
        assert neural_data_au[18][2837:2958].tolist() == np.array([-1, 2, 2, 1, 1, 1, 0, 0, -2, -6, -7, -6, -7,
                                                                            -5, 0, -1, -2, 1, -1, -5, -6, -11, -17, -15,
                                                                            -3, 8, 7, 3, 8, 11, 5, -2, -1, 6, 11, 10, 1, -7,
                                                                            -9, -7, -4, -2, 0, 5, 13, 21, 18, 6, -3, -3, 5,
                                                                            14, 12, 0, -8, -5, 0, -1, -6, -8, -3, 6, 16, 16,
                                                                            3, -7, -6, -3, -1, -1, -2, -5, -8, -5, 0, 4, 8,
                                                                            2, -6, -4, 0, -2, -6, -11, -11, -5, 3, 6, 3,
                                                                            -1, -2, -2, -2, -4, -8, -10, -9, -5, -7, -7, 2, 5,
                                                                            -1, -6, -3, 2, 2, -1, 0, 3, 0, -4, -2, 1, -1, -1, 1,
                                                                            -1, -6, -5, 3]).tolist()

    def test_import_open_ephys_channel_data_single_channel(self, all_channels_output, input_data):
        fs, neural_data_au, sync_data = all_channels_output
        folderpath, experiment, recording, source_processor = input_data

        assert neural_data_au[5].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=[5],
                                                        source_processor=source_processor)[1].tolist()
        assert neural_data_au[0].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=[0],
                                                        source_processor=source_processor)[1].tolist() #using channels=0 does not work, has to be a list

    def test_import_open_ephys_channel_data_multiple_channels(self, all_channels_output, input_data):
        fs, neural_data_au, sync_data = all_channels_output
        folderpath, experiment, recording, source_processor = input_data

        assert neural_data_au[21:26].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording,
                                                        channels=[21, 22, 23, 24, 25],
                                                        source_processor=source_processor)[1].tolist()
