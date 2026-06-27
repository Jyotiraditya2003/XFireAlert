def fuse_predictions(p_risk=None, p_det=None):
    """
    Adaptive multimodal fusion
    Handles missing modalities gracefully
    """

    # Case 1: No data
    if p_risk is None and p_det is None:
        return None

    # Case 2: Weather only
    if p_det is None:
        return p_risk

    # Case 3: Image only
    if p_risk is None:
        return p_det

    # Case 4: Both available

    # uncertainty-aware weighting
    # higher confidence → higher weight

    risk_weight = 0.55
    vision_weight = 0.45

    # If visible fire strong → trusting camera more
    if p_det > 0.75:
        risk_weight = 0.35
        vision_weight = 0.65

    # If weather extremely dangerous → trusting weather
    if p_risk > 0.85:
        risk_weight = 0.70
        vision_weight = 0.30

    return round(risk_weight*p_risk + vision_weight*p_det, 3)


def get_alert_level(p_final):

    if p_final is None:
        return "INSUFFICIENT DATA"

    if p_final > 0.8:
        return "CRITICAL"

    elif p_final > 0.6:
        return "DANGER"

    elif p_final > 0.4:
        return "HIGH"

    else:
        return "NORMAL"