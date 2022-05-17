import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import dirichlet
from sklearn.preprocessing import normalize
from .proper_losses import bs, log_loss


def force_p_to_simplex(p, crop=0):
    # bring the predictions back on the simplex if they move away
    p[p < 0] = crop
    p[p > 1] = 1 - crop
    return p / np.sum(p, axis=1, keepdims=True)


def minkowski_dist(x_test, x_train, p=2):
    """
    :param x_test: Predictions or logits in shape (rows, features).
    :param x_train: Predictions or logits in shape (rows, features).
    :param p: p=2 for Euclidean, p=1 for Manhattan distance.
    :return: Matrix of distances from every x_test to every x_train instance in shape (nr x_test rows, nr x_train rows)
    """
    return (np.abs(x_test[:, None] - x_train) ** p).sum(axis=2) ** (1 / p)


def kl(x_test, x_train):
    return np.sum(x_test[:, None] * np.log(x_test[:, None] / x_train), axis=2)


def dirichlet_log_kernel(x_test, x_train, h):
    all_weights = np.zeros((x_test.shape[0], x_train.shape[0]))
    x_test_precise = normalize(np.array(x_test, np.float64), norm="l1").T
    for pred_idx, pred in enumerate(x_train):
        alpha = pred / h + 1
        weights = dirichlet.logpdf(x_test_precise, alpha=alpha)
        all_weights[:, pred_idx] = weights
    return all_weights


def k_closest_uniform(distances, k):
    """
    :param distances: Distances from every test instance to every train instance in shape (nr test rows, nr train rows).
    :param k: Number of neighbors.
    :return: For every test instance weights 1/k to k-closest neighbors, weights 0 for others. Output matrix is in input shape.
    """
    k_closest_ids = np.argpartition(distances, k)[:, :k]
    weights = np.zeros(distances.shape)
    np.put_along_axis(weights, k_closest_ids, 1 / k, axis=1)
    return weights


def proportional_to_exp_kernel(kernel_values):
    """
    :return: Returns weights directly proportional to exp(kernel_values) (normalized to sum to 1 in every row).
    """
    exp_distances = np.exp(kernel_values - np.max(kernel_values, axis=1, keepdims=True))
    return normalize(exp_distances, norm="l1")


def proportional_to_kernel(kernel_values):
    """
    :return: Returns weights directly proportional to kernel_values (normalized to sum to 1 in every row).
    """
    return kernel_values / np.sum(kernel_values, axis=1, keepdims=True)


def weights_fun_example(x_test, x_train):
    return k_closest_uniform(minkowski_dist(x_test, x_train, p=2), k=100)
    # return proportional_to_dist(rbf_kernel(x_test, x_train))
    # return proportional_to_dist(kl(x_test, x_train))


def calibrate_preds_with_weighting_cv(x_train, p_train, y_train, x_test, p_test,
                                      weighting_funs_to_try, cv_loss, n_cv_folds, crop, batch_size):
    w_scores = find_weighting_fun_with_cv(x=x_train, p=x_train, y=y_train, weighting_funs_to_try=weighting_funs_to_try,
                                          batch_size=batch_size, crop=crop, n_cv_folds=n_cv_folds, loss=cv_loss)
    print(w_scores)
    w_fun_to_use = weighting_funs_to_try[np.argmin(w_scores)]
    cal_p_test = calibrate_preds_with_weighting(x_train=x_train, p_train=p_train, y_train=y_train,
                                                x_test=x_test, p_test=p_test,
                                                weights_fun=w_fun_to_use, batch_size=batch_size, crop=crop)

    cal_p_train = calibrate_preds_with_weighting(x_train=x_train, p_train=p_train, y_train=y_train,
                                                x_test=x_train, p_test=p_train,
                                                weights_fun=w_fun_to_use, batch_size=batch_size, crop=crop)

    return cal_p_test, cal_p_train, w_scores


def calibrate_preds_with_weighting(x_train, x_test, p_train, p_test, y_train, weights_fun, batch_size=32, crop=0):
    CE_estimates = calibration_error_at_points_with_weighting(x_train=x_train, x_test=x_test,
                                                              p_diff_y_train=p_train - y_train,
                                                              weights_fun=weights_fun, batch_size=batch_size)
    return force_p_to_simplex(p_test - CE_estimates, crop=crop)


def calibration_error_at_points_with_weighting(x_train, x_test, p_diff_y_train, weights_fun, batch_size=32):
    CE_estimates = np.zeros(x_test.shape)

    for x_idx in range(0, len(x_test), batch_size):
        batch_weights = weights_fun(x_test[x_idx:x_idx + batch_size], x_train)
        CE_estimates[x_idx:x_idx + batch_size] = np.dot(batch_weights, p_diff_y_train)
    return CE_estimates


def find_weighting_fun_with_cv(x, p, y, weighting_funs_to_try, batch_size=32, crop=0,
                               n_cv_folds=10, loss=bs):
    w_scores = [0] * len(weighting_funs_to_try)

    for w_idx, w_fun in enumerate(weighting_funs_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=0)

        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                cal_p_test = calibrate_preds_with_weighting(x_train=x_train, x_test=x_test,
                                                            p_train=p_train, p_test=p_test,
                                                            y_train=y_train, weights_fun=w_fun,
                                                            batch_size=batch_size, crop=crop)

                cv_scores.append(loss(y_test, cal_p_test))
            except:
                "Something wrong in cross-validation. Are parameters to try correct?"
                cv_scores.append(99999)

        w_scores[w_idx] = np.mean(cv_scores)
        print(np.mean(cv_scores))

    return w_scores


# cw w new implementation try

def calibrate_preds_with_cw_weighting_cv(x_train, p_train, y_train, x_test, p_test,
                                         cw_weighting_funs_to_try, cv_loss, n_cv_folds, crop, batch_size):
    w_scores = find_cw_weighting_fun_with_cv(x=x_train, p=x_train, y=y_train,
                                             cw_weighting_funs_to_try=cw_weighting_funs_to_try,
                                             batch_size=batch_size, crop=crop, n_cv_folds=n_cv_folds, loss=cv_loss)
    print(w_scores)
    w_fun_to_use = cw_weighting_funs_to_try[np.argmin(w_scores)]
    cal_p_test = calibrate_preds_with_cw_weighting(x_train=x_train, p_train=p_train, y_train=y_train,
                                                   x_test=x_test, p_test=p_test,
                                                   cw_weights_fun=w_fun_to_use, batch_size=batch_size, crop=crop)

    cal_p_train = calibrate_preds_with_cw_weighting(x_train=x_train, p_train=p_train, y_train=y_train,
                                                   x_test=x_train, p_test=p_train,
                                                   cw_weights_fun=w_fun_to_use, batch_size=batch_size, crop=crop)

    return cal_p_test, cal_p_train, w_scores


def cw_weighted_euc_knn(x_test, x_train, class_idx, k, class_weight, other_classes_weight):
    """
    Returns weights matrix in shape (nr rows x_test, nr rows x_train) for class 'class_idx'.
    Distances are found with weighted Euclidean distance where class 'class_idx' is multiplied with 'class_weight**2'
    and other classes with 'other_classes_weight**2'. Distances are turned into weights according to KNN with param 'k'.
    """
    weights = np.full(x_test.shape[1], other_classes_weight)
    weights[class_idx] = class_weight

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train * weights, np.zeros(x_train.shape[0]))
    neigh_ids = neigh.kneighbors(x_test * weights, return_distance=False)

    weights = np.zeros((x_test.shape[0], x_train.shape[0]))
    np.put_along_axis(weights, neigh_ids, 1 / k, axis=1)
    return weights


def calibrate_preds_with_cw_weighting(x_train, x_test, p_train, p_test, y_train, cw_weights_fun, batch_size=32, crop=0):
    CE_estimates = calibration_error_at_points_with_cw_weighting(x_train=x_train, x_test=x_test,
                                                                 p_diff_y_train=p_train - y_train,
                                                                 cw_weights_fun=cw_weights_fun, batch_size=batch_size)
    return force_p_to_simplex(p_test - CE_estimates, crop=crop)


def calibration_error_at_points_with_cw_weighting(x_train, x_test, p_diff_y_train, cw_weights_fun, batch_size=32):
    CE_estimates = np.zeros(x_test.shape)

    for x_idx in range(0, len(x_test), batch_size):
        for class_idx in range(x_train.shape[1]):
            class_i_batch_weights = cw_weights_fun(x_test[x_idx:x_idx + batch_size], x_train, class_idx=class_idx)
            CE_estimates[x_idx:x_idx + batch_size, class_idx] = np.dot(class_i_batch_weights,
                                                                       p_diff_y_train[:, class_idx])
    return CE_estimates


def find_cw_weighting_fun_with_cv(x, p, y, cw_weighting_funs_to_try, batch_size=32, crop=0,
                                  n_cv_folds=10, loss=bs):
    w_scores = [0] * len(cw_weighting_funs_to_try)

    for w_idx, w_fun in enumerate(cw_weighting_funs_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=0)

        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                cal_p_test = calibrate_preds_with_cw_weighting(x_train=x_train, x_test=x_test,
                                                               p_train=p_train, p_test=p_test,
                                                               y_train=y_train, cw_weights_fun=w_fun,
                                                               batch_size=batch_size, crop=crop)

                cv_scores.append(loss(y_test, cal_p_test))
            except:
                "Something wrong in cross-validation. Are parameters to try correct?"
                cv_scores.append(99999)

        w_scores[w_idx] = np.mean(cv_scores)
        print(np.mean(cv_scores))

    return w_scores


"""
# old knn
# old knn logit
# old knn mix with cw


### knn mix cw with classic

def calibrate_preds_with_knn_mix(p_train, y_train, p_test, k1, k2, threshold, crop=0):
    CE_estimates_cw = calibration_error_at_points_with_knn_cw(p_train, y_train, p_test, k=k1)
    CE_estimates = calibration_error_at_points_with_knn(p_train, y_train, p_test, k=k2)

    CE_estimates_combined = np.zeros(p_test.shape)
    CE_estimates_combined[p_test<threshold] = CE_estimates_cw[p_test<threshold]
    CE_estimates_combined[~(p_test<threshold)] = CE_estimates[~(p_test<threshold)]

    calibrated_preds = p_test - CE_estimates_combined
    calibrated_preds = force_p_to_simplex(calibrated_preds, crop=crop)
    return calibrated_preds

### knn

def calibrate_preds_with_knn(p_train, y_train, p_test, k=100, adjustment=None, n_calibrated_copies=0, distance_metric="euclidean", crop=0):
    # adjustment "multiplicative", "additive" or None (classic additive)
    CE_estimates = calibration_error_at_points_with_knn(p_train, y_train, p_test, k=k, adjustment=adjustment, n_calibrated_copies=n_calibrated_copies, distance_metric=distance_metric)
    calibrated_preds = p_test - CE_estimates
    calibrated_preds = force_p_to_simplex(calibrated_preds, crop=crop)

    return calibrated_preds

def calibration_error_at_points_with_knn(p_train, y_train, p_test, k=100, adjustment=None, n_calibrated_copies=0, distance_metric="euclidean"):
    from kood.shift_happens_adjustment_code.experiments.adjusters import BS_general_adjuster, LL_general_adjuster

    if distance_metric == "euclidean":
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(p_train, y_train)
        neigh_ids = neigh.kneighbors(p_test, return_distance=False)
    elif distance_metric == "kl":
        neigh_ids = []
        for p in p_test:
            neigh_ids.append(np.sum(p * np.log(p / p_train), axis=1).argsort()[:k])
        neigh_ids = np.array(neigh_ids)

    mean_p_s = np.mean(p_train[neigh_ids], axis=1)
    mean_y_s = np.mean(y_train[neigh_ids], axis=1)

    if adjustment == None:
        CE_estimates = mean_p_s - mean_y_s

        if n_calibrated_copies > 0:
            from scipy.stats import multinomial
            def mean_y_sampler(y_probabilities):
                return multinomial.rvs(k, y_probabilities, size=n_calibrated_copies) / k
            n_data = p_test.shape[0]
            n_classes = p_test.shape[1]
            calibrated_copies_mean_y_s = np.apply_along_axis(mean_y_sampler, 1, mean_p_s)

            calibrated_copies_CE_estimates = np.tile(mean_p_s, n_calibrated_copies).reshape(
                (n_data, n_calibrated_copies, n_classes)) - calibrated_copies_mean_y_s

            calibrated_copies_CE_estimate_lengths = np.linalg.norm(calibrated_copies_CE_estimates, axis=2)
            calibrated_copies_CE_estimate_length_quantiles = np.quantile(calibrated_copies_CE_estimate_lengths, q=1.0,
                                                                         axis=1)
            CE_estimate_lengths = np.linalg.norm(CE_estimates, axis=1)
            CE_estimates[CE_estimate_lengths <= calibrated_copies_CE_estimate_length_quantiles] = 0
    elif adjustment == "multiplicative":
        p_ll = []
        for i in range(len(mean_p_s)):
            neighbors = np.append(p_train[neigh_ids][i], [p_test[i]], axis=0)
            new_p = LL_general_adjuster(p_=neighbors, new_pi=mean_y_s[i],
                                        error=None, big_classes=None, small_classes=None)[0][-1]
            p_ll.append(new_p)
        p_ll = np.array(p_ll)
        CE_estimates = - (p_ll - p_test)
    elif adjustment == "additive":
        p_bs = []
        for i in range(len(mean_p_s)):
            neighbors = np.append(p_train[neigh_ids][i], [p_test[i]], axis=0)
            new_p = BS_general_adjuster(p_=neighbors, new_pi=mean_y_s[i],
                                        error=None, big_classes=None, small_classes=None)[0][-1]
            p_bs.append(new_p)
        p_bs = np.array(p_bs)
        CE_estimates = - (p_bs - p_test)

    return CE_estimates
    
def find_k_with_cv(p, y, k_s_to_try, n_cv_folds, adjustment=None, n_calibrated_copies=0, distance_metric="euclidean", crop=0, loss=bs):
    k_scores = [0] * len(k_s_to_try)

    for k_idx, k in enumerate(k_s_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=2)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            c_hat_test = calibrate_preds_with_knn(p_train, y_train, p_test, k=k, adjustment=adjustment, n_calibrated_copies=n_calibrated_copies, distance_metric=distance_metric, crop=crop)
            cv_scores.append(loss(y_test, c_hat_test))

        k_scores[k_idx] = np.mean(cv_scores)

    return k_scores
    
### kernel

from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from scipy.special import gamma

def dirichlet_kernel(p_train, p_test, h=1):
    weights = np.zeros((p_test.shape[0], p_train.shape[0]))

    for prediction_idx, prediction in enumerate(p_test):
        a = prediction / h + 1
        weights[prediction_idx, :] = gamma(a.sum()) / np.product(gamma(a)) * np.product(p_train ** (a - 1), axis=1)
    return weights.T

def calibrate_preds_with_kernel(p_train, y_train, p_test, crop=0, kernel=lambda x,y:rbf_kernel(x,y, gamma=1)):
    CE_estimates = calibration_error_at_points_with_kernel(p_train, y_train, p_test, kernel=kernel)
    calibrated_preds = p_test - CE_estimates
    calibrated_preds = force_p_to_simplex(calibrated_preds, crop=crop)
    return calibrated_preds


def calibration_error_at_points_with_kernel(p_train, y_train, p_test, kernel=lambda x,y:rbf_kernel(x,y, gamma=1)):
    neigh_weights = kernel(p_test, p_train)
    neigh_CE_estimates = p_train - y_train
    divider = np.sum(neigh_weights, axis=1).reshape((-1, 1))
    divider[divider == 0] = 9999999999
    return np.dot(neigh_weights, neigh_CE_estimates) / divider


def find_kernel_with_cv(p, y, kernels_s_to_try, n_cv_folds, crop=0, loss=bs):
    k_scores = [0] * len(kernels_s_to_try)

    for k_idx, kernel in enumerate(kernels_s_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=2)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            c_hat_test = calibrate_preds_with_kernel(p_train, y_train, p_test, crop=crop, kernel=kernel)
            cv_scores.append(loss(y_test, c_hat_test))

        k_scores[k_idx] = np.mean(cv_scores)

    return k_scores
    
### knn logit
def calibrate_preds_with_knn_logits(p_train, logits_train, y_train, p_test, logits_test, k=100, crop=0):
    CE_estimates = calibration_error_at_points_with_knn_logits(p_train, logits_train, y_train, logits_test, k=k)
    calibrated_preds = p_test - CE_estimates
    calibrated_preds = force_p_to_simplex(calibrated_preds, crop=crop)
    return calibrated_preds

def calibration_error_at_points_with_knn_logits(p_train, logits_train, y_train, logits_test, k=100):
    neigh = KNeighborsClassifier(n_neighbors=k, weights="distance")
    neigh.fit(logits_train, y_train)
    neigh_ids = neigh.kneighbors(logits_test, return_distance=False)

    mean_p_s = np.mean(p_train[neigh_ids], axis=1)
    mean_y_s = np.mean(y_train[neigh_ids], axis=1)
    return mean_p_s - mean_y_s

def find_k_with_cv_logit(p, y, logits, k_s_to_try, n_cv_folds, crop=0, loss=bs):
    k_scores = [0] * len(k_s_to_try)

    for k_idx, k in enumerate(k_s_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=2)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            logits_train, logits_test = logits[train_index], logits[test_index]
            y_train, y_test = y[train_index], y[test_index]

            c_hat_test = calibrate_preds_with_knn_logits(p_train, logits_train, y_train, p_test, logits_test, k=k, crop=crop)
            cv_scores.append(loss(y_test, c_hat_test))

        k_scores[k_idx] = np.mean(cv_scores)

    return k_scores
    
"""

"""
# old cw
# old cww

def calibrate_preds_with_knn_cw(p_train, y_train, p_test, k=100, crop=0):
    CE_estimates = calibration_error_at_points_with_knn_cw(p_train, y_train, p_test, k=k)
    calibrated_preds = p_test - CE_estimates
    return force_p_to_simplex(calibrated_preds, crop=crop)

def calibration_error_at_points_with_knn_cw(p_train, y_train, p_test, k=100):

    CE_estimates = np.zeros(p_test.shape)

    for class_idx in range(p_train.shape[1]):

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(p_train[:, class_idx, None], y_train[:, class_idx])
        neigh_ids = neigh.kneighbors(p_test[:, class_idx, None], return_distance=False)
        mean_p_s = np.mean(p_train[neigh_ids, class_idx], axis=1)
        mean_y_s = np.mean(y_train[neigh_ids, class_idx], axis=1)
        CE_estimates[:, class_idx] = mean_p_s - mean_y_s

    return CE_estimates

def find_k_with_cv_cw(p, y, k_s_to_try, n_cv_folds, crop=0, loss=bs):
    k_scores = [0] * len(k_s_to_try)

    for k_idx, k in enumerate(k_s_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=2)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            c_hat_test = calibrate_preds_with_knn_cw(p_train, y_train, p_test, k=k, crop=crop)
            cv_scores.append(loss(y_test, c_hat_test))

        k_scores[k_idx] = np.mean(cv_scores)

    return k_scores

### knn weighted cw

def calibrate_preds_with_knn_cw_weighted(p_train, y_train, p_test, k=100, weight=1, crop=0):
    CE_estimates = calibration_error_at_points_with_knn_cw_weighted(p_train, y_train, p_test, k=k, weight=weight)
    calibrated_preds = p_test - CE_estimates
    return force_p_to_simplex(calibrated_preds, crop=crop)

def calibration_error_at_points_with_knn_cw_weighted(p_train, y_train, p_test, k=100, weight=1):

    CE_estimates = np.zeros(p_test.shape)

    for class_idx in range(p_train.shape[1]):
        weights = np.ones(p_test.shape[1])
        weights[class_idx] = weight

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(p_train*weights, y_train)
        neigh_ids = neigh.kneighbors(p_test*weights, return_distance=False)
        mean_p_s = np.mean(p_train[neigh_ids, class_idx], axis=1)
        mean_y_s = np.mean(y_train[neigh_ids, class_idx], axis=1)
        CE_estimates[:, class_idx] = mean_p_s - mean_y_s

    return CE_estimates

def find_w_with_cv_cww(p, y, w_s_to_try, n_cv_folds, k, crop=0, loss=bs):
    w_scores = [0] * len(w_s_to_try)

    for w_idx, w in enumerate(w_s_to_try):
        cv_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=2)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            c_hat_test = calibrate_preds_with_knn_cw_weighted(p_train, y_train, p_test, k=k, weight=w, crop=crop)
            cv_scores.append(loss(y_test, c_hat_test))

        w_scores[w_idx] = np.mean(cv_scores)

    return w_scores

"""
