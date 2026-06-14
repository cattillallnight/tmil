import pickle
with open(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step02b_features_hybrid.pkl", 'rb') as f:
    records = pickle.load(f)
print(records[0].keys())
