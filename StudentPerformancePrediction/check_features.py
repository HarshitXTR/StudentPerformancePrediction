import joblib

print("="*50)

scaler = joblib.load("scaler.pkl")

print("Total Features")

print(len(scaler.feature_names_in_))

print()

print("Feature Names")

print("-"*50)

for feature in scaler.feature_names_in_:
    print(feature)

print("="*50)