import pickle
import sys

if len(sys.argv) != 2:
    print("Usage: python read_pkl.py <file.pkl>")
    sys.exit(1)

with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
    print(data)
