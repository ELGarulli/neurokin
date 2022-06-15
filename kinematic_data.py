from typing import Dict
import numpy as np


class KinematicData:

    def __init__(self, path, config_file):
        self.fs: float = 0.0
        self.config = self.load_config_file(config_file)
