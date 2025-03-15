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
        assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=True) == 0

    def test_time_to_sample_with_negative_timestamp(self):
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=-1, fs=40.2, is_t1=False, is_t2=False)

    def test_time_to_sample_with_fs_equal_0(self):
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1.75, fs=0, is_t1=False, is_t2=False)


    def test_time_to_sample_with_negative_fs(self):
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1, fs=-40.2, is_t1=False, is_t2=False)


class TestImportTDTChannelData:
    @pytest.fixture(autouse=True)
    def setup_paths(self, repo_root):
        self.folderpath = repo_root / "tests" / "test_data" / "TDT_test_data"
    def time_to_sample(self, timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
        key = get_key(timestamp)

        return_values = {get_key(0): 0,
                         get_key(14.61): 356352}

        return return_values.get(key, 'default_output')


    @pytest.fixture
    def all_channels_output(self, monkeypatch):
        monkeypatch.setattr(importing, 'time_to_sample', self.time_to_sample)
        fs, raw, stim_data, fs_sync = importing.import_tdt_channel_data(self.folderpath, t2=14.61)
        yield fs, raw, stim_data, fs_sync

    def test_import_tdt_channel_data_all_channels(self, all_channels_output):
        fs, raw, stim_data, fs_sync = all_channels_output

        assert fs == pytest.approx(24414.0625) and stim_data is None and fs_sync is None
        assert np.sum(raw) == pytest.approx(273678340.0)
        assert raw[3][238923] == pytest.approx(256.0)
        test_array = np.ones(1000)*256
        np.testing.assert_allclose(raw[5][252963:253963], test_array)

    def test_import_tdt_channel_data_single_channel(self, all_channels_output):
        fs, raw, stim_data, fs_sync = all_channels_output

        np.testing.assert_allclose(raw[3], importing.import_tdt_channel_data(self.folderpath, ch=3, t2=14.61)[1])
        np.testing.assert_allclose(raw[0], importing.import_tdt_channel_data(self.folderpath, ch=0, t2=14.61)[1])

    def test_import_tdt_channel_data_multiple_channels(self, all_channels_output):
        fs, raw, stim_data, fs_sync = all_channels_output

        np.testing.assert_allclose(raw[2:5], importing.import_tdt_channel_data(self.folderpath, ch=[2, 3, 4], t2=14.61)[1])


class TestImportOpenEphysChannelData:

    @pytest.fixture
    def input_data(self, repo_root):
        experiment = 'Record Node 102/experiment1'
        recording = 'recording1'
        source_processor = 'Rhythm_FPGA-100.0'
        folderpath = str(repo_root / "tests" / "test_data" / "OE_test_data")
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
        assert np.sum(neural_data_au) == -37636516985
        assert neural_data_au[18][2837:2847].tolist() == np.array([-1990, -2076, -2097, -2078,
                                                                   -2097, -2108, -2119, -2097,
                                                                   -2118, -2118]).tolist()

    def test_import_open_ephys_channel_data_single_channel(self, all_channels_output, input_data):
        fs, neural_data_au, sync_data = all_channels_output
        folderpath, experiment, recording, source_processor = input_data

        assert neural_data_au[5].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=[5],
                                                        source_processor=source_processor)[1].tolist()
        assert neural_data_au[0].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=0,
                                                        source_processor=source_processor)[1].tolist() #using channels=0 does not work, has to be a list

    def test_import_open_ephys_channel_data_multiple_channels(self, all_channels_output, input_data):
        fs, neural_data_au, sync_data = all_channels_output
        folderpath, experiment, recording, source_processor = input_data

        assert neural_data_au[21:26].tolist() == \
               importing.import_open_ephys_channel_data(folderpath, experiment, recording,
                                                        channels=[21, 22, 23, 24, 25],
                                                        source_processor=source_processor)[1].tolist()
