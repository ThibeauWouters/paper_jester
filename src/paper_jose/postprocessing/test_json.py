import json
import numpy as np

filename = "./figures/final_doppelgangers/E_sym_fixed_NEPs_ingo.json"

with open(filename, "r") as f:
    data = json.load(f)
    
print("data")
print(data)

for key, value in data.items():
    print(key)
    print(np.shape(value))