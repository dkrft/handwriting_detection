from hwdetect.utils import get_path

try:
    import cPickle as pickle
except:
    import pickle

def load(path=None):
    """if path is None, will default to 
    data/data_sets/1_pixel_labels/ariel_26-10_5959.pkl"""
    
    if path is None:
        path = get_path('hwdetect/data/data_sets/1_pixel_labels/ariel_26-10_5959.pkl')
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret