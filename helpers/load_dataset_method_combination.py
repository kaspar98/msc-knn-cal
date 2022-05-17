from . import config
import pickle
import numpy as np
from .temp_scaling import softmax


def load_dataset(dataset):
    with open(f"{config.LOGIT_DIR}/probs_{dataset}_logits.p", 'rb') as f:
        (logits_train, y_train), (logits_test, y_test) = pickle.load(f)
    p_train = softmax(logits_train)
    p_test = softmax(logits_test)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    y_test = np.eye(len(logits_train[0]))[y_test_flat]
    y_train = np.eye(len(logits_train[0]))[y_train_flat]
    return logits_train, p_train, y_train, y_train_flat, logits_test, p_test, y_test, y_test_flat

def load_result(dataset, method):
    with open(f"{config.RESULTS_DIR}/{dataset}/{method}.p", 'rb') as f:
        data = pickle.load(f)
    return data