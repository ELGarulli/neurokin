import numpy as np
import pytest
from neurokin.utils.kinematics import gait_params_basics


print(gait_params_basics.get_phase_at_max_amplitude(np.sin(np.radians(np.arange(360)))))