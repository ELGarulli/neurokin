import neurokin.utils.commons
from neurokin.utils.neural import importing
from numpy.typing import ArrayLike


class NeuralData:
    """
    This class represents the neural data recording of a single recording experiment.
    It allows to load neural data from multiple recording systems and to perform basic analysis on the raw data.
    """

    def __init__(self, path):
        self.path = path
        self.raw: ArrayLike
        self.fs: float
        self.sync_data: ArrayLike
        self.fs_sync: float
        self.pulses_count: int
        self.recording: int
        self.freq: ArrayLike[ArrayLike]
        self.pxx: ArrayLike[ArrayLike]
        self.recording_duration: float

    def load_tdt_data(self, sync_present: bool = False, stream_name="Wav1", stim_stream_name="Wav1", t1=0, t2=-1):
        self.fs, self.raw, self.sync_data, self.fs_sync = importing.import_tdt_channel_data(folderpath=self.path,
                                                                                            stream_name=stream_name,
                                                                                            stim_name=stim_stream_name,
                                                                                            sync_present=sync_present,
                                                                                            t1=t1,
                                                                                            t2=t2)

    def load_open_ephys(self, experiment, recording, channels=None, sync_present: bool = False, sync_ch: int = None,
                        source_processor: str = None):
        self.fs, self.raw, self.sync_data = importing.import_open_ephys_channel_data(folderpath=self.path,
                                                                                     experiment=experiment,
                                                                                     recording=recording,
                                                                                     channels=channels,
                                                                                     sync_present=sync_present,
                                                                                     sync_ch=sync_ch,
                                                                                     source_processor=source_processor)
        self.recording = recording


    def compute_recording_duration(self):
        self.recording_duration = self.raw.shape[1] / self.fs

    def set_pulses_count(self):
        return

    def pick_sync_data(self, idx):
        self.sync_data = self.sync_data[idx]
