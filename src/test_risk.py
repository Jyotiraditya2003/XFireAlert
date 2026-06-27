from risk_pipeline import predict_risk, explain_risk

weather = {
    "Temperature": 36,
    "RH": 18,
    "Ws": 20,
    "Rain": 0.0,
    "FFMC": 90.2,
    "DMC": 30.5,
    "DC": 110.4,
    "ISI": 7.1,
    "BUI": 35.2
}

prob = predict_risk(weather)
print("\nFire Risk Probability:", round(prob,3))

print("\nTop Contributing Factors:")
explanation = explain_risk(weather)

for feature, value in list(explanation.items())[:5]:
    print(feature, ":", round(value,4))

