"""
AutoRiskML Metrics Module
Performance metrics for risk models: PSI, CSI, KS, Gini, AUC
"""

import math
from typing import Dict, List, Optional, Tuple


def compute_psi(
    baseline_dist: List[float],
    current_dist: List[float]
) -> float:
    """
    Compute Population Stability Index (PSI)
    
    PSI < 0.1: No significant population shift
    0.1 <= PSI < 0.2: Moderate shift
    PSI >= 0.2: Significant shift (retrain recommended!)
    
    Args:
        baseline_dist: Baseline distribution (e.g., training data)
        current_dist: Current distribution (e.g., production data)
    
    Returns:
        PSI value
    """
    psi = 0.0
    
    for base_pct, curr_pct in zip(baseline_dist, current_dist):
        # Avoid log(0)
        base_pct = max(base_pct, 0.0001)
        curr_pct = max(curr_pct, 0.0001)
        
        psi += (curr_pct - base_pct) * math.log(curr_pct / base_pct)
    
    return psi


def compute_csi(
    baseline_stats: Dict[str, float],
    current_stats: Dict[str, float]
) -> float:
    """
    Compute Characteristic Stability Index (CSI)
    
    Similar to PSI but for feature characteristics (mean, std, etc.)
    
    Args:
        baseline_stats: Baseline statistics
        current_stats: Current statistics
    
    Returns:
        CSI value
    """
    # Simple implementation: compare mean and std
    baseline_mean = baseline_stats.get('mean', 0)
    baseline_std = baseline_stats.get('std', 1)
    current_mean = current_stats.get('mean', 0)
    current_std = current_stats.get('std', 1)
    
    # Standardized difference
    mean_diff = abs(current_mean - baseline_mean) / max(baseline_std, 0.0001)
    std_ratio = abs(current_std / max(baseline_std, 0.0001) - 1)
    
    csi = mean_diff + std_ratio
    return csi


def compute_ks_statistic(
    y_true: List[int],
    y_pred_proba: List[float]
) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov (KS) statistic
    
    KS measures the separation between good and bad distributions
    Higher KS = better model discrimination
    
    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities
    
    Returns:
        (ks_statistic, ks_threshold)
    """
    # Sort by predicted probability
    sorted_pairs = sorted(zip(y_pred_proba, y_true), reverse=True)
    
    n_total = len(y_true)
    n_bad = sum(y_true)
    n_good = n_total - n_bad
    
    if n_bad == 0 or n_good == 0:
        return 0.0, 0.5
    
    # Cumulative distributions
    cum_bad = 0
    cum_good = 0
    max_ks = 0.0
    ks_threshold = 0.5
    
    for prob, label in sorted_pairs:
        if label == 1:
            cum_bad += 1
        else:
            cum_good += 1
        
        # KS = max difference between cumulative distributions
        bad_rate = cum_bad / n_bad
        good_rate = cum_good / n_good
        ks = abs(bad_rate - good_rate)
        
        if ks > max_ks:
            max_ks = ks
            ks_threshold = prob
    
    return max_ks, ks_threshold


def compute_auc(
    y_true: List[int],
    y_pred_proba: List[float]
) -> float:
    """
    Compute Area Under ROC Curve (AUC)
    
    AUC = 0.5: Random model
    AUC > 0.7: Acceptable
    AUC > 0.8: Good
    AUC > 0.9: Excellent (check for data leakage!)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        AUC value
    """
    # Simple implementation using rank statistics
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Rank all samples by predicted probability
    ranked = sorted(zip(y_pred_proba, y_true), reverse=True)
    
    # Sum of ranks for positive class
    rank_sum = sum(i + 1 for i, (_, label) in enumerate(ranked) if label == 1)
    
    # AUC via Mann-Whitney U statistic
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    return auc


def compute_gini(auc: float) -> float:
    """
    Compute Gini coefficient from AUC
    
    Gini = 2 * AUC - 1
    
    Gini measures inequality in model predictions
    Higher Gini = better discrimination
    
    Args:
        auc: AUC value
    
    Returns:
        Gini coefficient
    """
    return 2 * auc - 1


def compute_brier_score(
    y_true: List[int],
    y_pred_proba: List[float]
) -> float:
    """
    Compute Brier score (calibration metric)
    
    Lower Brier score = better calibrated probabilities
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Brier score
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    
    brier = sum((p - y) ** 2 for y, p in zip(y_true, y_pred_proba)) / n
    return brier


def compute_lift(
    y_true: List[int],
    y_pred_proba: List[float],
    n_deciles: int = 10
) -> List[float]:
    """
    Compute lift chart
    
    Lift shows how much better the model is than random
    Lift > 1 = model is better than random
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_deciles: Number of deciles
    
    Returns:
        List of lift values per decile
    """
    n = len(y_true)
    overall_rate = sum(y_true) / n if n > 0 else 0
    
    # Sort by predicted probability (descending)
    sorted_pairs = sorted(zip(y_pred_proba, y_true), reverse=True)
    
    # Compute lift per decile
    decile_size = n // n_deciles
    lifts = []
    
    for i in range(n_deciles):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_deciles - 1 else n
        
        decile_labels = [label for _, label in sorted_pairs[start_idx:end_idx]]
        decile_rate = sum(decile_labels) / len(decile_labels) if decile_labels else 0
        
        lift = decile_rate / overall_rate if overall_rate > 0 else 1.0
        lifts.append(lift)
    
    return lifts


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, int]:
    """
    Compute confusion matrix
    
    Returns:
        Dictionary with TP, TN, FP, FN
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp + tn) / len(y_true) if len(y_true) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


def interpret_psi(psi: float) -> str:
    """Interpret PSI value"""
    if psi < 0.1:
        return "No significant shift (stable)"
    elif psi < 0.2:
        return "Moderate shift (monitor closely)"
    else:
        return "Significant shift (retrain recommended!)"


def interpret_ks(ks: float) -> str:
    """Interpret KS statistic"""
    if ks < 0.2:
        return "Poor discrimination"
    elif ks < 0.4:
        return "Fair discrimination"
    elif ks < 0.6:
        return "Good discrimination"
    else:
        return "Excellent discrimination"


def interpret_auc(auc: float) -> str:
    """Interpret AUC value"""
    if auc < 0.6:
        return "Poor (almost random)"
    elif auc < 0.7:
        return "Fair"
    elif auc < 0.8:
        return "Acceptable"
    elif auc < 0.9:
        return "Good"
    else:
        return "Excellent (check for data leakage!)"


# Export public API
__all__ = [
    'compute_psi',
    'compute_csi',
    'compute_ks_statistic',
    'compute_auc',
    'compute_gini',
    'compute_brier_score',
    'compute_lift',
    'compute_confusion_matrix',
    'interpret_psi',
    'interpret_ks',
    'interpret_auc'
]
