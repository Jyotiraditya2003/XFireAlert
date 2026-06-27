def choose_model(p_risk=None, mobilenet_prob=None):
    """
    Intelligent model selection
    Works even if weather OR image data missing
    """

    # If no image available
    if mobilenet_prob is None:
        return "weather_only"

    # IMAGE ONLY MODEL
    if p_risk is None:

        if mobilenet_prob > 0.90:
            return "yolo"

        if mobilenet_prob > 0.50:
            return "vgg"

        return "mobilenet"

    # HYBRID MODE

    # smoke plume / distant wildfire
    if mobilenet_prob > 0.75 and p_risk > 0.65:
        return "vgg"

    # visible nearby flames
    if mobilenet_prob > 0.85 and p_risk < 0.6:
        return "yolo"

    # uncertain case
    if 0.40 < mobilenet_prob <= 0.75:
        return "vgg"

    # low evidence
    return "mobilenet"