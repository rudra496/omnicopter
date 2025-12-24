"""
Power Proxy Calculation Utilities

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import numpy as np


def power_proxy(rotor_thrusts, k_t=1.0):
    """
    Compute power proxy from rotor thrusts
    
    Args:
        rotor_thrusts: Array of rotor thrust values
        k_t: Thrust coefficient
        
    Returns:
        power: Power proxy value
    """
    # Power proxy: P ∝ Σ(T_i^(3/2))
    power = np.sum(np.abs(rotor_thrusts) ** 1.5) / k_t
    
    return power


def compute_energy_savings(power_expert, power_baseline):
    """
    Compute energy savings percentage
    
    Args:
        power_expert: Power consumption with expert policy
        power_baseline: Power consumption with baseline
        
    Returns:
        savings_percent: Energy savings as percentage
    """
    savings_percent = (power_baseline - power_expert) / power_baseline * 100
    
    return savings_percent


def compute_power_statistics(powers):
    """
    Compute power consumption statistics
    
    Args:
        powers: Array of power values
        
    Returns:
        stats: Dictionary of statistics
    """
    stats = {
        'mean': np.mean(powers),
        'std': np.std(powers),
        'min': np.min(powers),
        'max': np.max(powers),
        'median': np.median(powers),
        'q25': np.percentile(powers, 25),
        'q75': np.percentile(powers, 75)
    }
    
    return stats