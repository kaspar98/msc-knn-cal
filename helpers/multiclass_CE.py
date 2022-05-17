from .binnings import EqualSizeBinning
import numpy as np

def confidence_equal_size_ece(p, y, n_bins=15):
    binning = EqualSizeBinning(p=p.max(axis=1), y=y[range(len(y)), p.argmax(axis=1)], c=None, n_bins=n_bins)
    return {"ece_abs": binning.ECE_abs,
            "ece_abs_db": binning.ECE_abs_debiased,
            "ece_sq": binning.ECE_square,
            "ece_sq_db": binning.ECE_square_debiased}


def classwise_equal_size_ece(p, y, n_bins=15, threshold=0):
    n_classes = p.shape[1]

    ece_abs_s = []
    ece_abs_db_s = []
    ece_sq_s = []
    ece_sq_db_s = []

    for data_class in range(n_classes):
        p_class = p[:, data_class]
        y_class = y[:, data_class]
        binning = EqualSizeBinning(p=p_class[p_class>threshold], y=y_class[p_class>threshold], c=None, n_bins=n_bins)
        ece_abs_s.append(binning.ECE_abs)
        ece_abs_db_s.append(binning.ECE_abs_debiased)
        ece_sq_s.append(binning.ECE_square)
        ece_sq_db_s.append(binning.ECE_square_debiased)

    return {"ece_abs": np.mean(ece_abs_s),
            "ece_abs_db": np.mean(ece_abs_db_s),
            "ece_sq": np.mean(ece_sq_s),
            "ece_sq_db": np.mean(ece_sq_db_s)}

def CE_estimation_distance(mean_p, mean_y, d=1):
    return np.power(np.sum(np.power(np.abs(mean_y - mean_p), d)), 1/d)


def CE_distance(p, c, d=1):
    return np.mean(CE_distances(p, c, d))


def CE_distances(p, c, d=1):
    return np.power(np.sum(np.power(np.abs(p - c), d), axis=1), 1/d)