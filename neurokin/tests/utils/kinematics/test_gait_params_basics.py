import numpy as np
import pytest
from neurokin.utils.kinematics import gait_params_basics


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


# class TestComparePhase():
#
#    def get_phase_at_max_amplitude(self, input_signal):
#        key = get_key(input_signal)
#
#        def raise_exception():
#            raise TypeError(f"Invalid input: {input_signal}")
#
#        return_values = {
#            get_key(np.sin(np.radians(np.arange(360))).tolist()): 270.0,
#            get_key([]): raise_exception,
#            get_key(["a"]): raise_exception,
#            get_key(["a", "b"]): raise_exception,
#            get_key([1]): 360.0,
#            get_key([1, 2]): -90.0
#        }
#        result = return_values.get(key, "default_output")
#
#        return result
#
#    def test_compare_phase(self, mocker):
#        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
#                     side_effect=self.get_phase_at_max_amplitude)
#        compare_phase = gait_params_basics.compare_phase
#
#        with pytest.raises(TypeError):
#            compare_phase([], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], [])
#        with pytest.raises(TypeError):
#            compare_phase(['a'], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], ['a'])
#        with pytest.raises(TypeError):
#            compare_phase(['a', 'b'], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], ['a', 'b'])
#        assert compare_phase([1, 2], [1]) == pytest.approx(-450.0)
#        assert compare_phase(np.sin(np.radians(np.arange(360))).tolist(), [1, 2]) == pytest.approx(360.0)

class TestComparePhase:

    def get_phase_at_max_amplitude(self, input_signal):
        key = get_key(input_signal)

        def raise_exception():
            raise TypeError(f"Invalid input: {input_signal}")

        return_values = {
            get_key(np.sin(np.radians(np.arange(360))).tolist()): 270.0,
            get_key([]): raise_exception,
            get_key(["a"]): raise_exception,
            get_key(["a", "b"]): raise_exception,
            get_key([1]): 360.0,
            get_key([1, 2]): 360.0,
        }
        result = return_values.get(key, "default_output")

        return result

    def test_compare_phase_with_empty_first_argument(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        with pytest.raises(TypeError):
            self.compare_phase([], [1])

    def test_compare_phase_with_empty_second_argument(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        with pytest.raises(TypeError):
            self.compare_phase([1], [])

    def test_compare_phase_valid_input(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        assert self.compare_phase([1, 2], [1]) == pytest.approx(0.0)

    def test_compare_phase_with_numpy_array(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        assert self.compare_phase(np.sin(np.radians(np.arange(360))).tolist(), [1, 2]) == pytest.approx(-90.0)
