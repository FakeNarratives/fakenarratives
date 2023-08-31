import logging
import pickle


def read_pkl(fname):
    try:
        with open(fname, "rb") as pklfile:
            data = pickle.load(pklfile)
    except Exception as e:
        logging.error(f"Cannot load {fname}. {e}")

    return data
