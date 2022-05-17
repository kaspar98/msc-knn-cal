from .load_dataset_method_combination import load_dataset
from .temp_scaling import softmax
import numpy as np
from .config import PRECOMPUTED_RESULTS_DIR

def iop_precomp(dataset):

    logits_train, p_train, y_train, y_train_flat, logits_test, p_test, y_test, y_test_flat = load_dataset(dataset)

    preds_te = np.load(f"{PRECOMPUTED_RESULTS_DIR}/iop/{dataset}/scores.npy")
    logits_te = np.load(f"{PRECOMPUTED_RESULTS_DIR}/iop/{dataset}/logits.npy")

    i_te = []
    for x in logits_test[:, 0]:
        index = list(logits_te[:, 0]).index(x)
        i_te.append(index)
    preds_te = preds_te[i_te]

    return softmax(preds_te), np.ones(p_train.shape), None
