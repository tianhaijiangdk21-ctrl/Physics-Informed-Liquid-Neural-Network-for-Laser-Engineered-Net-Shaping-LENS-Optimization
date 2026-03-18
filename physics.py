"""
Helper functions for physics-informed loss (simplified algebraic proxies).
Replace with actual PDE solvers for more accurate physics.
"""
import torch

def fourier_proxy(P, V, F, eta):
    """
    Approximate heat conduction residual based on process parameters.
    Returns a tensor of residuals (batch,).
    """
    # Simple relationship: temperature should be proportional to P/(V*F)
    # In normalized form.
    heat_input = P / (V * F + 1e-6)
    return heat_input

def thermo_proxy(P, V, F, eta):
    """
    Approximate thermo-mechanical residual.
    """
    # Residual stress proxy: proportional to P (laser power)
    return P