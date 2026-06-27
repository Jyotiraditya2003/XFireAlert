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

p = predict_risk(weather)
print("Fire Risk Probability:", p)

exp = explain_risk(weather)
print("\nTop Factors:")
for k,v in list(exp.items())[:5]:
    print(k, ":", round(v,4))
