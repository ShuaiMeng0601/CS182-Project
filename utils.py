# utils.py
import pickle

def save_processed(states, actions, path):
    data = {"states": states, "actions": actions}
    with open(path, "wb") as f:
        pickle.dump(data, f)
