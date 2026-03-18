"""
Evaluation metrics: R², MAE, MAPE, and Python implementation of CUI.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE = mean(|y_true - y_pred| / |y_true|) * 100%"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero; add small epsilon
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_cui_from_measurements(widths, heights):
    """
    Compute CUI from arrays of measured widths and heights.
    widths: list of widths in mm
    heights: list of top heights in mm
    Returns CUI (scalar)
    """
    mu_w = np.mean(widths)
    sigma_w = np.std(widths, ddof=1)
    mu_h = np.mean(heights)
    sigma_h = np.std(heights, ddof=1)
    if mu_w == 0 or mu_h == 0:
        return np.nan
    cui = 1 - np.sqrt((sigma_w/mu_w)**2 + (sigma_h/mu_h)**2)
    return max(0, min(1, cui))

# For completeness, a function that simulates CUI from image would be too complex here.
# The actual CUI calculation from images is provided in Supplementary Note S2 (MATLAB).