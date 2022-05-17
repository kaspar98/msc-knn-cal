import numpy as np

from sklearn.metrics import log_loss

def bs(p, y):
    return np.mean(np.sum((p - y) ** 2, axis=1))