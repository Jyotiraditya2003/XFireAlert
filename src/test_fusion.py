from fusion_engine import fuse_predictions, get_alert_level

# example outputs from two models
p_risk = 0.99
p_det = 0.88

p_final = fuse_predictions(p_risk, p_det)
alert = get_alert_level(p_final)

print("Final Probability:", p_final)
print("Alert Level:", alert)