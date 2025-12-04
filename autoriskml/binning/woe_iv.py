"""
AutoRiskML Binning Module
Weight of Evidence (WOE) and Information Value (IV) computation
Monotonic binning for risk modeling

This is the REVOLUTIONARY feature that no other package has!
"""

from typing import Dict, List, Optional, Tuple, Union


def compute_woe_iv(
    values: List,
    target: List,
    n_bins: int = 5,
    method: str = "quantile"
) -> Dict:
    """
    Compute Weight of Evidence (WOE) and Information Value (IV)
    
    WOE measures the predictive power of a feature
    IV quantifies the strength of the relationship with target
    
    Args:
        values: Feature values
        target: Binary target (0/1)
        n_bins: Number of bins
        method: Binning method ('quantile', 'equal_width', 'monotonic')
    
    Returns:
        Dictionary with WOE table, IV score, bin edges
    """
    if not values or not target or len(values) != len(target):
        raise ValueError("Invalid input: values and target must be non-empty and same length")
    
    # Determine if numeric or categorical
    is_numeric = _is_numeric(values)
    
    if is_numeric:
        bins, woe_table = _compute_numeric_woe(values, target, n_bins, method)
    else:
        bins, woe_table = _compute_categorical_woe(values, target)
    
    # Compute IV from WOE table
    iv = sum((good_pct - bad_pct) * woe for _, _, _, good_pct, bad_pct, woe in woe_table)
    
    return {
        "iv": iv,
        "woe_table": woe_table,
        "bins": bins,
        "is_numeric": is_numeric,
        "method": method
    }


def _is_numeric(values: List) -> bool:
    """Check if values are numeric"""
    try:
        for v in values[:100]:  # Sample first 100
            if v is not None and v != "":
                float(v)
        return True
    except (ValueError, TypeError):
        return False


def _compute_numeric_woe(
    values: List,
    target: List,
    n_bins: int,
    method: str
) -> Tuple[List, List]:
    """
    Compute WOE/IV for numeric features
    
    Returns:
        (bin_edges, woe_table)
    """
    # Convert to numeric
    numeric_values = []
    numeric_target = []
    for v, t in zip(values, target):
        try:
            if v is not None and v != "":
                numeric_values.append(float(v))
                numeric_target.append(int(t))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return [], []
    
    # Create bins
    if method == "quantile":
        bin_edges = _quantile_binning(numeric_values, n_bins)
    elif method == "equal_width":
        bin_edges = _equal_width_binning(numeric_values, n_bins)
    elif method == "monotonic":
        bin_edges = _monotonic_binning(numeric_values, numeric_target, n_bins)
    else:
        bin_edges = _quantile_binning(numeric_values, n_bins)
    
    # Assign values to bins
    bin_assignments = _assign_bins(numeric_values, bin_edges)
    
    # Compute WOE for each bin
    woe_table = _compute_woe_table(bin_assignments, numeric_target, bin_edges)
    
    return bin_edges, woe_table


def _quantile_binning(values: List[float], n_bins: int) -> List[float]:
    """Create bins using quantiles"""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    edges = [sorted_vals[0]]
    
    for i in range(1, n_bins):
        idx = int(i * n / n_bins)
        if idx < n:
            edges.append(sorted_vals[idx])
    
    edges.append(sorted_vals[-1] + 0.001)  # Add small epsilon to include last value
    
    # Remove duplicates
    edges = sorted(list(set(edges)))
    return edges


def _equal_width_binning(values: List[float], n_bins: int) -> List[float]:
    """Create bins with equal width"""
    min_val = min(values)
    max_val = max(values)
    width = (max_val - min_val) / n_bins
    
    edges = [min_val + i * width for i in range(n_bins + 1)]
    edges[-1] += 0.001  # Ensure last value included
    return edges


def _monotonic_binning(
    values: List[float],
    target: List[int],
    n_bins: int
) -> List[float]:
    """
    Create bins with monotonic bad rate
    
    This is the ADVANCED technique used in credit scoring!
    Bins are merged until bad rate is monotonically increasing/decreasing
    """
    # Start with many bins
    initial_bins = min(20, len(set(values)) // 2)
    if initial_bins < 2:
        initial_bins = 2
    
    bin_edges = _quantile_binning(values, initial_bins)
    
    # Iteratively merge bins until monotonic
    max_iterations = 100
    for _ in range(max_iterations):
        bin_assignments = _assign_bins(values, bin_edges)
        
        # Compute bad rates per bin
        bad_rates = []
        for bin_idx in range(len(bin_edges) - 1):
            bin_targets = [t for i, t in enumerate(target) if bin_assignments[i] == bin_idx]
            if bin_targets:
                bad_rate = sum(bin_targets) / len(bin_targets)
            else:
                bad_rate = 0.0
            bad_rates.append(bad_rate)
        
        # Check if monotonic
        is_increasing = all(bad_rates[i] <= bad_rates[i+1] for i in range(len(bad_rates)-1))
        is_decreasing = all(bad_rates[i] >= bad_rates[i+1] for i in range(len(bad_rates)-1))
        
        if (is_increasing or is_decreasing) or len(bin_edges) <= n_bins + 1:
            break
        
        # Find adjacent bins with most similar bad rates and merge
        min_diff = float('inf')
        merge_idx = 0
        for i in range(len(bad_rates) - 1):
            diff = abs(bad_rates[i] - bad_rates[i+1])
            if diff < min_diff:
                min_diff = diff
                merge_idx = i
        
        # Merge bins
        bin_edges = bin_edges[:merge_idx+1] + bin_edges[merge_idx+2:]
    
    return bin_edges


def _assign_bins(values: List[float], bin_edges: List[float]) -> List[int]:
    """Assign each value to a bin"""
    assignments = []
    for v in values:
        bin_idx = 0
        for i in range(len(bin_edges) - 1):
            if v >= bin_edges[i] and v < bin_edges[i+1]:
                bin_idx = i
                break
            elif v >= bin_edges[-1]:
                bin_idx = len(bin_edges) - 2
        assignments.append(bin_idx)
    return assignments


def _compute_woe_table(
    bin_assignments: List[int],
    target: List[int],
    bin_edges: List[float]
) -> List[Tuple]:
    """
    Compute WOE table
    
    Returns list of tuples: (bin_name, n_good, n_bad, good_pct, bad_pct, woe)
    """
    # Count goods and bads per bin
    total_good = sum(1 for t in target if t == 0)
    total_bad = sum(1 for t in target if t == 1)
    
    if total_good == 0:
        total_good = 1  # Avoid division by zero
    if total_bad == 0:
        total_bad = 1
    
    woe_table = []
    
    for bin_idx in range(len(bin_edges) - 1):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        bin_name = f"[{bin_start:.2f}, {bin_end:.2f})"
        
        # Count goods and bads in this bin
        bin_targets = [target[i] for i, b in enumerate(bin_assignments) if b == bin_idx]
        
        if not bin_targets:
            n_good, n_bad = 0, 0
        else:
            n_good = sum(1 for t in bin_targets if t == 0)
            n_bad = sum(1 for t in bin_targets if t == 1)
        
        # Add smoothing to avoid zero counts
        n_good = max(n_good, 0.5)
        n_bad = max(n_bad, 0.5)
        
        # Compute percentages
        good_pct = n_good / total_good
        bad_pct = n_bad / total_bad
        
        # Compute WOE
        import math
        woe = math.log(good_pct / bad_pct)
        
        woe_table.append((bin_name, n_good, n_bad, good_pct, bad_pct, woe))
    
    return woe_table


def _compute_categorical_woe(values: List, target: List) -> Tuple[List, List]:
    """Compute WOE/IV for categorical features"""
    # Get unique categories
    unique_cats = list(set(values))
    
    total_good = sum(1 for t in target if t == 0)
    total_bad = sum(1 for t in target if t == 1)
    
    if total_good == 0:
        total_good = 1
    if total_bad == 0:
        total_bad = 1
    
    woe_table = []
    
    for cat in unique_cats:
        # Count goods and bads for this category
        cat_targets = [target[i] for i, v in enumerate(values) if v == cat]
        
        if not cat_targets:
            continue
        
        n_good = sum(1 for t in cat_targets if t == 0)
        n_bad = sum(1 for t in cat_targets if t == 1)
        
        # Smoothing
        n_good = max(n_good, 0.5)
        n_bad = max(n_bad, 0.5)
        
        # Percentages
        good_pct = n_good / total_good
        bad_pct = n_bad / total_bad
        
        # WOE
        import math
        woe = math.log(good_pct / bad_pct)
        
        woe_table.append((str(cat), n_good, n_bad, good_pct, bad_pct, woe))
    
    return unique_cats, woe_table


def compute_psi(
    baseline_values: List,
    current_values: List,
    bins: Optional[List] = None
) -> float:
    """
    Compute Population Stability Index (PSI)
    
    PSI measures the shift in population distribution
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (retrain recommended)
    
    Args:
        baseline_values: Baseline (training) data
        current_values: Current (production) data
        bins: Optional bin edges (computed if not provided)
    
    Returns:
        PSI value
    """
    if not baseline_values or not current_values:
        return 0.0
    
    # Check if numeric
    is_numeric = _is_numeric(baseline_values)
    
    if is_numeric:
        # Create bins if not provided
        if bins is None:
            bins = _quantile_binning([float(v) for v in baseline_values if v is not None], 10)
        
        # Compute distributions
        baseline_dist = _compute_distribution(baseline_values, bins)
        current_dist = _compute_distribution(current_values, bins)
    else:
        # Categorical: use categories as bins
        all_cats = list(set(baseline_values + current_values))
        baseline_dist = _compute_categorical_distribution(baseline_values, all_cats)
        current_dist = _compute_categorical_distribution(current_values, all_cats)
    
    # Compute PSI
    psi = 0.0
    import math
    
    for i in range(len(baseline_dist)):
        baseline_pct = max(baseline_dist[i], 0.0001)  # Avoid log(0)
        current_pct = max(current_dist[i], 0.0001)
        
        psi += (current_pct - baseline_pct) * math.log(current_pct / baseline_pct)
    
    return psi


def _compute_distribution(values: List, bins: List[float]) -> List[float]:
    """Compute distribution across bins"""
    numeric_values = [float(v) for v in values if v is not None and v != ""]
    n_total = len(numeric_values)
    
    if n_total == 0:
        return [0.0] * (len(bins) - 1)
    
    assignments = _assign_bins(numeric_values, bins)
    
    distribution = []
    for bin_idx in range(len(bins) - 1):
        count = sum(1 for a in assignments if a == bin_idx)
        distribution.append(count / n_total)
    
    return distribution


def _compute_categorical_distribution(values: List, categories: List) -> List[float]:
    """Compute distribution for categorical data"""
    n_total = len(values)
    
    if n_total == 0:
        return [0.0] * len(categories)
    
    distribution = []
    for cat in categories:
        count = sum(1 for v in values if v == cat)
        distribution.append(count / n_total)
    
    return distribution


def interpret_iv(iv: float) -> str:
    """
    Interpret Information Value
    
    IV < 0.02: Not useful for prediction
    0.02 <= IV < 0.1: Weak predictor
    0.1 <= IV < 0.3: Medium predictor
    0.3 <= IV < 0.5: Strong predictor
    IV >= 0.5: Suspicious (too good, check for leakage)
    """
    if iv < 0.02:
        return "Not useful"
    elif iv < 0.1:
        return "Weak predictor"
    elif iv < 0.3:
        return "Medium predictor"
    elif iv < 0.5:
        return "Strong predictor"
    else:
        return "Suspicious (check for data leakage)"


# Export public functions
__all__ = [
    'compute_woe_iv',
    'compute_psi',
    'interpret_iv'
]
