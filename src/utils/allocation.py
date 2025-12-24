"""
Null-Space Allocation Utilities

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import numpy as np


class AllocationModel:
    """Null-space allocation for omnidirectional UAV"""
    
    def __init__(self, n_rotors=8):
        """
        Initialize allocation model
        
        Args:
            n_rotors: Number of rotors
        """
        self.n_rotors = n_rotors
        self.allocation_matrix = None
        
    def compute_allocation_matrix(self, vehicle_params):
        """
        Compute allocation matrix from vehicle parameters
        
        Args:
            vehicle_params: Dictionary of vehicle parameters
            
        Returns:
            allocation_matrix: (6, n_rotors) allocation matrix
        """
        # Placeholder implementation
        # In practice, this would compute the actual allocation matrix
        # based on rotor positions, orientations, etc.
        self.allocation_matrix = np.random.randn(6, self.n_rotors)
        return self.allocation_matrix
    
    def compute_nullspace_basis(self):
        """
        Compute null-space basis vectors
        
        Returns:
            nullspace_basis: (n_rotors, 2) null-space basis
        """
        # Compute SVD of allocation matrix
        U, S, Vt = np.linalg.svd(self.allocation_matrix)
        
        # Null-space basis from last columns of V
        nullspace_basis = Vt[-2:, :].T
        
        return nullspace_basis
    
    def allocate_control(self, wrench_desired, nullspace_coeffs):
        """
        Allocate control with null-space optimization
        
        Args:
            wrench_desired: Desired 6D wrench
            nullspace_coeffs: [z1, z2] null-space coefficients
            
        Returns:
            rotor_thrusts: Allocated rotor thrusts
        """
        # Pseudo-inverse solution
        A_pinv = np.linalg.pinv(self.allocation_matrix)
        thrust_baseline = A_pinv @ wrench_desired
        
        # Null-space projection
        N = self.compute_nullspace_basis()
        thrust_nullspace = N @ nullspace_coeffs
        
        # Final allocation
        rotor_thrusts = thrust_baseline + thrust_nullspace
        
        return rotor_thrusts