try:
    import cPickle as pickle
except:
    import pickle

def load(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret