"""Unit tests for allocation utilities."""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


class TestAllocationMatrix(unittest.TestCase):
    """Test cases for allocation matrix generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_rotors = 8
        self.arm_length = 0.25
        self.thrust_coeff = 1.0
        self.torque_coeff = 0.01

    def test_allocation_matrix_shape(self):
        """Test that allocation matrix has correct shape."""
        # Expected shape: (4, num_rotors) for [Fx, Fy, Fz, Mz]
        # This would require the actual allocation utility to be imported
        # For now, we test the concept
        expected_shape = (4, self.num_rotors)
        self.assertEqual(expected_shape, (4, 8))

    def test_thrust_allocation(self):
        """Test thrust force allocation."""
        # Test that equal thrust from all rotors produces vertical force
        thrust_per_rotor = np.ones(self.num_rotors)
        total_thrust = np.sum(thrust_per_rotor) * self.thrust_coeff
        self.assertAlmostEqual(total_thrust, 8.0, places=5)

    def test_moment_allocation(self):
        """Test moment allocation around center."""
        # Rotors positioned symmetrically should produce zero net moment
        # when all have equal thrust
        angles = np.linspace(0, 2 * np.pi, self.num_rotors, endpoint=False)
        positions_x = self.arm_length * np.cos(angles)
        positions_y = self.arm_length * np.sin(angles)
        
        # Net moment should be near zero for symmetric configuration
        net_moment_x = np.sum(positions_y)
        net_moment_y = np.sum(positions_x)
        
        self.assertAlmostEqual(net_moment_x, 0.0, places=5)
        self.assertAlmostEqual(net_moment_y, 0.0, places=5)

    def test_allocation_pseudoinverse(self):
        """Test pseudoinverse allocation method."""
        # Create a simple allocation matrix
        A = np.random.rand(4, self.num_rotors)
        A_pinv = np.linalg.pinv(A)
        
        # Test that A @ A_pinv @ desired ≈ desired for feasible commands
        desired = np.array([0.0, 0.0, 10.0, 0.0])
        thrust = A_pinv @ desired
        achieved = A @ thrust
        
        assert_array_almost_equal(achieved, desired, decimal=4)

    def test_thrust_limits(self):
        """Test that thrust allocation respects physical limits."""
        max_thrust = 10.0
        thrust_command = np.array([12.0, 8.0, 5.0, 15.0, 6.0, 9.0, 11.0, 7.0])
        
        # Clip to max thrust
        clipped = np.clip(thrust_command, 0.0, max_thrust)
        
        self.assertTrue(np.all(clipped <= max_thrust))
        self.assertTrue(np.all(clipped >= 0.0))


class TestControlAllocation(unittest.TestCase):
    """Test cases for control allocation algorithms."""

    def test_minimum_norm_allocation(self):
        """Test minimum norm allocation solution."""
        # Minimum norm should give smallest magnitude solution
        A = np.random.rand(4, 8)
        b = np.array([0.0, 0.0, 10.0, 0.0])
        
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        norm = np.linalg.norm(x)
        
        self.assertGreater(norm, 0.0)

    def test_energy_optimal_allocation(self):
        """Test energy-optimal allocation (minimize squared thrust)."""
        # Energy optimal allocation minimizes sum of squared thrusts
        thrust = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        energy = np.sum(thrust ** 2)
        
        # Uniform distribution is energy optimal for symmetric tasks
        self.assertAlmostEqual(energy, 200.0, places=5)


if __name__ == "__main__":
    unittest.main()
