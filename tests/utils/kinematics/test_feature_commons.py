import numpy as np
import pandas as pd
import pytest
from neurokin.utils.features_extraction import commons


class TestGetAngle:
    def test_compute_angle_2d_right_angle(self):
        # For 2D, we provide three points: a = (0,0), b = (1,0), c = (1,1).
        # The angle at point b is 90° (π/2 radians).
        vectors = np.array([[0, 0, 1, 0, 1, 1]])
        angles = commons.compute_angle(vectors)
        expected = np.array([np.pi / 2])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_3d_right_angle(self):
        # For 3D, we use points: a = (0,0,0), b = (1,0,0), c = (1,1,0)
        # This also produces an angle of π/2.
        vectors = np.array([[0, 0, 0, 1, 0, 0, 1, 1, 0]])
        angles = commons.compute_angle(vectors)
        expected = np.array([np.pi / 2])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_2d_straight_line(self):
        # When the points are collinear, the angle should be 0.
        vectors = np.array([[0, 0, 1, 0, 2, 0]])
        angles = commons.compute_angle(vectors)
        expected = np.array([0.0])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_multiple_rows(self):
        # Testing with multiple rows.
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],  # right angle (π/2)
            [0, 0, 2, 0, 2, 2]  # right angle (π/2)
        ])
        angles = commons.compute_angle(vectors)
        expected = np.array([np.pi / 2, np.pi / 2])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_from_dataframe_2d(self):
        # Use a pandas DataFrame for a 2D input.
        df = pd.DataFrame({
            'a1': [0],
            'a2': [0],
            'b1': [1],
            'b2': [0],
            'c1': [1],
            'c2': [1]
        })
        angles = commons.compute_angle(df.values)
        expected = np.array([np.pi / 2])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_from_dataframe_3d(self):
        # Use a pandas DataFrame for a 3D input.
        df = pd.DataFrame({
            'a1': [0],
            'a2': [0],
            'a3': [0],
            'b1': [1],
            'b2': [0],
            'b3': [0],
            'c1': [1],
            'c2': [1],
            'c3': [0]
        })
        angles = commons.compute_angle(df.values)
        expected = np.array([np.pi / 2])
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_compute_angle_invalid_shape(self):
        # Passing an array with an invalid shape (not 6 or 9 columns) should result in an error.
        # Depending on how compute_angle is written, this might raise an IndexError or a NameError.
        vectors = np.array([[1, 2, 3, 4]])  # shape (1, 4)
        with pytest.raises((IndexError, NameError)):
            commons.compute_angle(vectors)

    def test_compute_angle_zero_norm(self):
        # When the first vector (b - a) is a zero vector (e.g. a and b are identical),
        # the computation leads to a division by zero resulting in a NaN angle.
        vectors = np.array([[0, 0, 0, 0, 1, 1]])
        angles = commons.compute_angle(vectors)
        # Check that the computed angle is NaN
        assert np.isnan(angles[0])


class TestGetAngleCorrelation:
    def test_compute_angle_correlation_single_row(self):
        # Use a simple 2D example.
        vectors = np.array([[0, 0, 1, 0, 1, 1]])
        # compute_angle returns an array of one angle, so np.corrcoef should return a (1, 1) matrix with 1.
        expected_corr = np.corrcoef(commons.compute_angle(vectors))
        result = commons.compute_angle_correlation(vectors)
        np.testing.assert_allclose(result, expected_corr, rtol=1e-5)

    def test_compute_angle_correlation_multiple_rows(self):
        # Use two rows which, through compute_angle, both yield the same angle (π/2).
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 2, 0, 2, 2]
        ])
        expected_corr = np.corrcoef(commons.compute_angle(vectors))
        result = commons.compute_angle_correlation(vectors)
        np.testing.assert_allclose(result, expected_corr, rtol=1e-5)


class TestComputeAngleVelocity:
    def test_multiple_rows(self):
        # Test with two rows.
        # Row 1: a = (0,0), b = (1,0), c = (1,1) => angle = π/2.
        # Row 2: a = (0,0), b = (1,0), c = (2,0) => angle = 0.
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 2, 0]
        ])
        angles = commons.compute_angle(vectors)
        expected = np.gradient(angles, 1)
        result = commons.compute_angle_velocity(vectors)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestComputeAngleAcceleration:
    def test_two_rows(self):
        # Provide two rows to ensure np.gradient can be computed.
        # Row 1: a = (0,0), b = (1,0), c = (1,1) produces a computed angle.
        # Row 2: a = (0,0), b = (1,0), c = (2,0) produces a different angle.
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 2, 0]
        ])
        # Compute the expected acceleration using np.gradient on the velocity.
        velocity = commons.compute_angle_velocity(vectors)
        expected_acc = np.gradient(velocity, 1)
        result = commons.compute_angle_acceleration(vectors)
        np.testing.assert_allclose(result, expected_acc, rtol=1e-5)


class TestGetPhase:
    def test_get_phase_with_1_point(self):
        with pytest.raises(IndexError):
            commons.compute_angle_phase(np.array([1]))

    def test_get_phase_with_2D_signal(self):
        np.random.seed(42)
        dummy_vectors = np.random.rand(60, 6)
        assert np.allclose(commons.compute_angle_phase(dummy_vectors)[:10],
                           np.array([0., -2.657708, 0.915085, -1.619009, 0.347632, 0.284068, -1.204167, -2.042451,
                                     -2.848609, 1.015215]))

    def test_get_phase_with_3D_signal(self):
        np.random.seed(42)
        dummy_vectors = np.random.rand(60, 9)
        assert np.allclose(commons.compute_angle_phase(dummy_vectors)[:10],
                           np.array(
                               [0., 1.47265759, 1.23881217, -2.4170581, 2.39504264, 0.03731343, 1.49577231, -1.81348321,
                                1.32230544, 3.12603581]))


class TestGetPhaseAtMaxAmplitude:

    def test_constant_angles(self):
        # Two identical rows will yield a constant angles series.
        # For example, using the 2D input [0, 0, 1, 0, 1, 1] produces an angle of π/2.
        # A constant series has a dominant DC component. Its phase is computed as np.angle(c_transform[0])
        # which is 0. Then the function adjusts this to 360.
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1]
        ])
        expected = 360.0
        result = commons.compute_phase_at_max_amplitude(vectors)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nonconstant_angles(self):
        # Provide two rows with different vectors so that the angles array is non-constant.
        # Row 1: [0, 0, 1, 0, 1, 1] produces an angle of π/2.
        # Row 2: [0, 0, 1, 0, 0, 1] produces an angle of approximately 2.3562 rad (135°).
        # In a two-element FFT, the DC term (index 0) will still likely dominate,
        # but we calculate the expected phase using the same logic as the function.
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1]
        ])
        # Replicate the internal logic to compute the expected phase.
        angles = commons.compute_angle(vectors)
        c_transform = np.fft.fft(angles)
        r_transform = abs(c_transform) ** 2
        freq_of_interest = np.argmax(r_transform)
        phase = np.angle(c_transform[freq_of_interest], deg=True)
        phase = phase if phase > 0 else 360 + phase
        expected = phase
        result = commons.compute_phase_at_max_amplitude(vectors)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_get_phase_at_max_amplitude_with_empty_array(self):
        with pytest.raises(IndexError):
            commons.compute_phase_at_max_amplitude(np.array([]))

    def test_get_phase_at_max_amplitude_with_vector_input(self):
        vectors = np.array([[0, 0, 1, 0, 1, 1]])
        result = commons.compute_phase_at_max_amplitude(vectors)
        assert result == pytest.approx(360.0)

    def test_get_phase_at_max_amplitude_with_1_point(self):
        vectors = np.array([[0, 0, 1, 0, 1, 1]])
        result = commons.compute_phase_at_max_amplitude(vectors)
        assert result == pytest.approx(360.0)

    def test_get_phase_at_max_amplitude_with_2_points(self):
        vectors = np.array([
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1]
        ])
        result = commons.compute_phase_at_max_amplitude(vectors)
        assert result == pytest.approx(360.0)

    def test_get_phase_at_max_amplitude_with_sin_input(self, monkeypatch):
        monkeypatch.setattr(commons, "compute_angle", lambda vectors: np.sin(np.radians(np.arange(360))))
        # The input here is a dummy array with the proper shape.
        dummy_vectors = np.empty((360, 6))
        result = commons.compute_phase_at_max_amplitude(dummy_vectors)
        assert result == pytest.approx(270.0, rel=1e-2)


class TestComputeSpeed:
    def test_speed_linear(self):
        # Create a simple linear trajectory DataFrame with two coordinates.
        # For column 'x': [0, 1, 2, 3, 4] --> np.gradient returns [1, 1, 1, 1, 1]
        # For column 'y': [0, 2, 4, 6, 8] --> np.gradient returns [2, 2, 2, 2, 2]
        # Thus, at each time point, speed = norm([1, 2]) = sqrt(1^2 + 2^2) = sqrt(5)
        df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 2, 4, 6, 8]
        })
        expected_speed = np.full(5, np.sqrt(5))
        result = commons.compute_speed(df)
        np.testing.assert_allclose(result, expected_speed, rtol=1e-5)


class TestComputeVelocity:
    def test_velocity_linear(self):
        # When compute_velocity is applied to a Series (as it is via df.apply in compute_speed),
        # it receives a 1D array. For a linear series, the gradient is constant.
        s = pd.Series([0, 1, 2, 3, 4])
        expected_velocity = np.gradient(s.values, 1)
        result = commons.compute_velocity(s)
        np.testing.assert_allclose(result, expected_velocity, rtol=1e-5)


class TestComputeAcceleration:
    def test_acceleration_linear(self):
        # For a linear series, the velocity is constant and thus the acceleration is zero.
        s = pd.Series([0, 1, 2, 3, 4])
        velocity = np.gradient(s.values, 1)
        expected_acceleration = np.gradient(velocity, 1)
        result = commons.compute_acceleration(s)
        np.testing.assert_allclose(result, expected_acceleration, rtol=1e-5)


class TestComputeTangAcceleration:
    def test_tang_acceleration_linear(self):
        # Create a linear trajectory DataFrame with two coordinates.
        # As shown in TestComputeSpeed, the speed is constant (sqrt(5) at every time point)
        # so its gradient (i.e. tangential acceleration) should be zero.
        df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 2, 4, 6, 8]
        })
        speed = commons.compute_speed(df)
        expected_tang_acc = np.gradient(speed, 1)
        result = commons.compute_tang_acceleration(df)
        np.testing.assert_allclose(result, expected_tang_acc, rtol=1e-5)
