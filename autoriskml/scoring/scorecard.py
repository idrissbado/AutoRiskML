"""
AutoRiskML Scorecard Module
Convert logistic regression models to credit scorecards
"""

import math
from typing import Dict, List, Optional, Tuple, Any


def generate_scorecard(
    model_coef: Dict[str, float],
    woe_tables: Dict[str, List[Tuple]],
    base_score: int = 600,
    pdo: int = 20,
    base_odds: float = 50.0
) -> Dict[str, Any]:
    """
    Generate credit scorecard from logistic regression model
    
    Scorecard converts WOE to points using PDO (Points to Double Odds)
    
    Formula:
    Score = base_score + (pdo / ln(2)) * (ln(odds) - ln(base_odds))
    Points = (pdo / ln(2)) * coefficient * WOE
    
    Args:
        model_coef: Feature coefficients from logistic regression
        woe_tables: WOE tables per feature
        base_score: Base score (e.g., 600 for FICO-like scale)
        pdo: Points to double odds (e.g., 20 means +20 points = odds * 2)
        base_odds: Base odds ratio (e.g., 50 = 50:1 good:bad)
    
    Returns:
        Scorecard dictionary with feature points tables
    """
    # Calculate scaling factors
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    
    scorecard = {
        'base_score': base_score,
        'pdo': pdo,
        'base_odds': base_odds,
        'factor': factor,
        'offset': offset,
        'features': {}
    }
    
    # Convert each feature's WOE to points
    for feature, coef in model_coef.items():
        if feature == 'intercept':
            # Intercept contributes to base score
            scorecard['intercept_contribution'] = factor * coef
            continue
        
        if feature not in woe_tables:
            continue
        
        woe_table = woe_tables[feature]
        points_table = []
        
        for bin_info in woe_table:
            # bin_info = (bin, n_good, n_bad, good_pct, bad_pct, woe)
            bin_label = bin_info[0]
            woe = bin_info[5]
            
            # Calculate points for this bin
            points = round(factor * coef * woe)
            
            points_table.append({
                'bin': bin_label,
                'woe': woe,
                'points': points
            })
        
        scorecard['features'][feature] = {
            'coefficient': coef,
            'points_table': points_table
        }
    
    return scorecard


def score_with_scorecard(
    data: Dict[str, Any],
    scorecard: Dict[str, Any],
    woe_tables: Dict[str, List[Tuple]]
) -> Dict[str, Any]:
    """
    Score a single observation using scorecard
    
    Args:
        data: Feature values
        scorecard: Scorecard from generate_scorecard()
        woe_tables: WOE tables for bin assignment
    
    Returns:
        Score and breakdown
    """
    # Start with base score
    total_points = scorecard['offset']
    
    # Add intercept contribution
    if 'intercept_contribution' in scorecard:
        total_points += scorecard['intercept_contribution']
    
    points_breakdown = []
    
    # Calculate points for each feature
    for feature, value in data.items():
        if feature not in scorecard['features']:
            continue
        
        feature_scorecard = scorecard['features'][feature]
        points_table = feature_scorecard['points_table']
        
        # Find which bin this value falls into
        if feature in woe_tables:
            bin_label = _find_bin(value, woe_tables[feature])
            
            # Find points for this bin
            for bin_points in points_table:
                if bin_points['bin'] == bin_label:
                    points = bin_points['points']
                    total_points += points
                    
                    points_breakdown.append({
                        'feature': feature,
                        'value': value,
                        'bin': bin_label,
                        'points': points
                    })
                    break
    
    # Convert total points to final score
    final_score = round(total_points)
    
    # Calculate probability from score
    log_odds = (final_score - scorecard['offset']) / scorecard['factor']
    odds = math.exp(log_odds)
    probability = odds / (1 + odds)
    
    # Determine risk tier
    risk_tier = _get_risk_tier(final_score, scorecard['base_score'])
    
    return {
        'score': final_score,
        'probability': probability,
        'risk_tier': risk_tier,
        'points_breakdown': points_breakdown
    }


def batch_score_with_scorecard(
    data_list: List[Dict[str, Any]],
    scorecard: Dict[str, Any],
    woe_tables: Dict[str, List[Tuple]]
) -> List[Dict[str, Any]]:
    """
    Score multiple observations with scorecard
    
    Args:
        data_list: List of feature dictionaries
        scorecard: Scorecard
        woe_tables: WOE tables
    
    Returns:
        List of score results
    """
    return [
        score_with_scorecard(data, scorecard, woe_tables)
        for data in data_list
    ]


def scorecard_to_dict(scorecard: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert scorecard to serializable dictionary
    
    Args:
        scorecard: Scorecard object
    
    Returns:
        Dictionary representation
    """
    return {
        'base_score': scorecard['base_score'],
        'pdo': scorecard['pdo'],
        'base_odds': scorecard['base_odds'],
        'factor': scorecard['factor'],
        'offset': scorecard['offset'],
        'intercept_contribution': scorecard.get('intercept_contribution', 0),
        'features': {
            feature: {
                'coefficient': info['coefficient'],
                'points_table': info['points_table']
            }
            for feature, info in scorecard['features'].items()
        }
    }


def scorecard_to_markdown(scorecard: Dict[str, Any]) -> str:
    """
    Convert scorecard to markdown table for reporting
    
    Args:
        scorecard: Scorecard object
    
    Returns:
        Markdown string
    """
    md = f"# Credit Scorecard\n\n"
    md += f"**Base Score:** {scorecard['base_score']}\n"
    md += f"**PDO:** {scorecard['pdo']} (Points to Double Odds)\n"
    md += f"**Base Odds:** {scorecard['base_odds']}\n\n"
    
    for feature, info in scorecard['features'].items():
        md += f"## {feature}\n"
        md += f"**Coefficient:** {info['coefficient']:.4f}\n\n"
        md += "| Bin | WOE | Points |\n"
        md += "|-----|-----|--------|\n"
        
        for bin_points in info['points_table']:
            bin_label = bin_points['bin']
            woe = bin_points['woe']
            points = bin_points['points']
            md += f"| {bin_label} | {woe:.4f} | {points:+d} |\n"
        
        md += "\n"
    
    return md


def _find_bin(value: Any, woe_table: List[Tuple]) -> str:
    """
    Find which bin a value belongs to
    
    Args:
        value: Feature value
        woe_table: WOE table
    
    Returns:
        Bin label
    """
    # For numeric values, woe_table bins are ranges like "(-inf, 10.5]"
    # For categorical, bins are category names
    
    if isinstance(value, (int, float)):
        # Numeric: find range
        for bin_info in woe_table:
            bin_label = bin_info[0]
            if _value_in_range(value, bin_label):
                return bin_label
    else:
        # Categorical: exact match
        for bin_info in woe_table:
            bin_label = bin_info[0]
            if str(value) == str(bin_label):
                return bin_label
    
    # Default: return first bin (or "Missing")
    return woe_table[0][0] if woe_table else "Missing"


def _value_in_range(value: float, range_str: str) -> bool:
    """
    Check if value is in range like "(-inf, 10.5]"
    
    Args:
        value: Numeric value
        range_str: Range string
    
    Returns:
        True if value in range
    """
    try:
        # Parse range string: "(-inf, 10.5]" or "[10.5, 20.0)"
        range_str = range_str.strip()
        
        # Extract bounds
        left_inclusive = range_str[0] == '['
        right_inclusive = range_str[-1] == ']'
        
        bounds = range_str[1:-1].split(',')
        left_bound = float(bounds[0].strip()) if 'inf' not in bounds[0] else float('-inf')
        right_bound = float(bounds[1].strip()) if 'inf' not in bounds[1] else float('inf')
        
        # Check if value in range
        left_ok = (value > left_bound) if not left_inclusive else (value >= left_bound)
        right_ok = (value < right_bound) if not right_inclusive else (value <= right_bound)
        
        return left_ok and right_ok
    
    except:
        # If parsing fails, assume value not in range
        return False


def _get_risk_tier(score: int, base_score: int) -> str:
    """
    Map score to risk tier
    
    Args:
        score: Credit score
        base_score: Base score (midpoint)
    
    Returns:
        Risk tier label
    """
    if score >= base_score + 100:
        return "Very Low Risk"
    elif score >= base_score + 50:
        return "Low Risk"
    elif score >= base_score:
        return "Medium Risk"
    elif score >= base_score - 50:
        return "High Risk"
    else:
        return "Very High Risk"


def explain_score(
    score_result: Dict[str, Any],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate reason codes for a score
    
    Args:
        score_result: Result from score_with_scorecard()
        top_n: Number of top factors to return
    
    Returns:
        List of reason codes
    """
    breakdown = score_result['points_breakdown']
    
    # Sort by absolute points contribution
    sorted_factors = sorted(
        breakdown,
        key=lambda x: abs(x['points']),
        reverse=True
    )
    
    # Take top N
    top_factors = sorted_factors[:top_n]
    
    # Format reason codes
    reasons = []
    for i, factor in enumerate(top_factors, 1):
        feature = factor['feature']
        value = factor['value']
        bin_label = factor['bin']
        points = factor['points']
        
        impact = "positive" if points > 0 else "negative"
        
        reasons.append({
            'rank': i,
            'feature': feature,
            'value': value,
            'bin': bin_label,
            'points': points,
            'impact': impact,
            'reason': f"{feature} in range {bin_label} ({impact} impact: {points:+d} points)"
        })
    
    return reasons


# Export public API
__all__ = [
    'generate_scorecard',
    'score_with_scorecard',
    'batch_score_with_scorecard',
    'scorecard_to_dict',
    'scorecard_to_markdown',
    'explain_score'
]
