import os
import pickle
import time
import numpy as np

from helpers import config
from helpers.iop_precomp import iop_precomp
from helpers.proper_losses import bs, log_loss
from helpers.weighting_calibrators import k_closest_uniform, proportional_to_kernel, \
    minkowski_dist, calibrate_preds_with_weighting_cv, rbf_kernel, \
    dirichlet_log_kernel, proportional_to_exp_kernel, kl
from helpers.calibration_trees import calibrate_preds_with_random_forest_cv
from helpers.dir_scaling import calibrate_with_ms_pretuned, calibrate_with_dir_pretuned
from helpers.vec_scaling import calibrate_preds_with_vs
from helpers.temp_scaling import calibrate_preds_with_temps, TemperatureScaling
from helpers.dec_cal import calibrate_with_deccal
from helpers.load_dataset_method_combination import load_dataset


def run_method(method_with_cv_params, dataset_name, method_name):
    method = method_with_cv_params.pop("method")
    cv_hyperparam_names = method_with_cv_params.pop("cv_hyperparam_names", None)

    cal_p_test, cal_p_train, cv_param_scores = method(**method_with_cv_params)

    if cv_param_scores == None:
        opt_cv_param = None
        n_cv_folds = None
    else:
        opt_cv_param = cv_hyperparam_names[np.argmin(cv_param_scores)]
        n_cv_folds = method_with_cv_params["n_cv_folds"]

    data = {"method_name": method_name, "dataset_name": dataset_name,
            "cv_param": opt_cv_param, "all_cv_params": cv_hyperparam_names, "all_cv_scores": cv_param_scores,
            "n_cv_folds": n_cv_folds,
            "cal_p_test": cal_p_test,
            "cal_p_train": cal_p_train}
    print(f"method_name {method_name}")
    print(f"cv_param {opt_cv_param}")

    with open(f'{config.RESULTS_DIR + "/" + dataset_name}/{method_name}.p', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cal_p_test


def run(dataset_name, method_name):
    # Make results directory
    dataset_results_dir = config.RESULTS_DIR + "/" + dataset_name
    if not os.path.exists(dataset_results_dir):
        os.makedirs(dataset_results_dir)

    # Load dataset
    logits_train, p_train, y_train, y_train_flat, logits_test, p_test, y_test, y_test_flat = load_dataset(dataset_name)

    # Fit temperature scaling on data for composition methods
    ts = TemperatureScaling()
    ts.fit(logits_train, y_train_flat)
    p_test_TS = ts.predict(logits_test)
    p_train_TS = ts.predict(logits_train)
    print(f"TempS temp: {ts.temp}, TempS BS: {bs(p=p_test_TS, y=y_test)}")

    k_s_to_try = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    gammas_to_try = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4]
    depths_to_try = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    batch_size = 300
    n_cv_folds = 10

    if logits_train.shape[1] == 100:
        crop = 1e-6
    elif logits_train.shape[1] == 10:
        crop = 1e-4

    # Select method
    methods = {
        "uncalibrated": {"method": lambda p_test, p_train: (p_test, p_train, None),
                         "p_test": p_test,
                         "p_train": p_train
                         },
        "dir_ODIR_pt": {"method": calibrate_with_dir_pretuned,
                        "p_train": p_train,
                        "y_train": y_train,
                        "p_test": p_test,
                        "dataset": dataset_name
                        },
        "ms_ODIR_pt": {"method": calibrate_with_ms_pretuned,
                       "x_train": logits_train,
                       "y_train": y_train,
                       "x_test": logits_test,
                       "dataset": dataset_name
                       },
        "vecS": {"method": calibrate_preds_with_vs,
                 "logits_train": logits_train,
                 "y_train": y_train,
                 "logits_test": logits_test,
                 },
        "tempS": {"method": calibrate_preds_with_temps,
                  "logits_train": logits_train,
                  "y_train": y_train,
                  "logits_test": logits_test,
                  },
        "knnTS_euc_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                            "x_train": p_train_TS,
                            "p_train": p_train_TS,
                            "y_train": y_train,
                            "x_test": p_test_TS,
                            "p_test": p_test_TS,
                            "weighting_funs_to_try": [
                                lambda test, train, k=k: k_closest_uniform(minkowski_dist(test, train, p=2), k)
                                for k in k_s_to_try],
                            "cv_hyperparam_names": k_s_to_try,
                            "cv_loss": log_loss,
                            "n_cv_folds": n_cv_folds,
                            "crop": crop,
                            "batch_size": batch_size
                            },
        "knnTS_kl_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                           "x_train": p_train_TS,
                           "p_train": p_train_TS,
                           "y_train": y_train,
                           "x_test": p_test_TS,
                           "p_test": p_test_TS,
                           "weighting_funs_to_try": [
                               lambda test, train, k=k: k_closest_uniform(kl(test, train), k)
                               for k in k_s_to_try],
                           "cv_hyperparam_names": k_s_to_try,
                           "cv_loss": log_loss,
                           "n_cv_folds": n_cv_folds,
                           "crop": crop,
                           "batch_size": batch_size
                           },
        "knn_kl_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                         "x_train": p_train,
                         "p_train": p_train,
                         "y_train": y_train,
                         "x_test": p_test,
                         "p_test": p_test,
                         "weighting_funs_to_try": [
                             lambda test, train, k=k: k_closest_uniform(kl(test, train), k)
                             for k in k_s_to_try],
                         "cv_hyperparam_names": k_s_to_try,
                         "cv_loss": log_loss,
                         "n_cv_folds": n_cv_folds,
                         "crop": crop,
                         "batch_size": batch_size
                         },
        "kernelTS_RBF_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                               "x_train": p_train_TS,
                               "p_train": p_train_TS,
                               "y_train": y_train,
                               "x_test": p_test_TS,
                               "p_test": p_test_TS,
                               "weighting_funs_to_try": [
                                   lambda test, train, g=g: proportional_to_kernel(rbf_kernel(test, train, gamma=g))
                                   for g in gammas_to_try],
                               "cv_hyperparam_names": gammas_to_try,
                               "cv_loss": log_loss,
                               "n_cv_folds": n_cv_folds,
                               "crop": crop,
                               "batch_size": batch_size
                               },
        "kernelTS_DIR_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                               "x_train": p_train_TS,
                               "p_train": p_train_TS,
                               "y_train": y_train,
                               "x_test": p_test_TS,
                               "p_test": p_test_TS,
                               "weighting_funs_to_try": [
                                   lambda test, train, g=g: proportional_to_exp_kernel(
                                       dirichlet_log_kernel(test, train, h=g))
                                   for g in gammas_to_try],
                               "cv_hyperparam_names": gammas_to_try,
                               "cv_loss": log_loss,
                               "n_cv_folds": n_cv_folds,
                               "crop": crop,
                               "batch_size": batch_size
                               },
        "kernel_DIR_cv_ll": {"method": calibrate_preds_with_weighting_cv,
                             "x_train": p_train,
                             "p_train": p_train,
                             "y_train": y_train,
                             "x_test": p_test,
                             "p_test": p_test,
                             "weighting_funs_to_try": [
                                 lambda test, train, g=g: proportional_to_exp_kernel(
                                     dirichlet_log_kernel(test, train, h=g))
                                 for g in gammas_to_try],
                             "cv_hyperparam_names": gammas_to_try,
                             "cv_loss": log_loss,
                             "n_cv_folds": n_cv_folds,
                             "crop": crop,
                             "batch_size": batch_size
                             },
        "rfTS_cv_ll": {"method": calibrate_preds_with_random_forest_cv,
                       "p_train": p_train_TS,
                       "y_train": y_train,
                       "p_test": p_test_TS,
                       "n_trees": 250,
                       "depths_to_try": depths_to_try,
                       "cv_hyperparam_names": [f"depth={d}" for d in depths_to_try],
                       "cv_loss": log_loss,
                       "n_cv_folds": n_cv_folds,
                       "crop": crop,
                       "seed": 0
                       },
        "rf_cv_ll": {"method": calibrate_preds_with_random_forest_cv,
                     "p_train": p_train,
                     "y_train": y_train,
                     "p_test": p_test,
                     "n_trees": 250,
                     "depths_to_try": depths_to_try,
                     "cv_hyperparam_names": [f"depth={d}" for d in depths_to_try],
                     "cv_loss": log_loss,
                     "n_cv_folds": n_cv_folds,
                     "crop": crop,
                     "seed": 0
                     },
        "dec2TS": {"method": calibrate_with_deccal,
                   "p_train": p_train_TS,
                   "y_train": y_train,
                   "p_test": p_test_TS,
                   "y_test": y_test,
                   "crop": crop,
                   "epochs": 1000,
                   "calib_steps": 100,
                   "num_action": 2
                   },
        "iop_diag": {"method": iop_precomp,
                     "dataset": dataset_name
                     }
    }
    selected_method = methods[method_name]

    cal_p = run_method(method_with_cv_params=selected_method, dataset_name=dataset_name, method_name=method_name)
    print(f"BS: {bs(cal_p, y_test)}")
    print("Done")


if __name__ == '__main__':
    for dataset_name in [
        "densenet40_c10",
        "densenet40_c100",
        "resnet_wide32_c10",
        "resnet_wide32_c100",
        "resnet110_c10",
        "resnet110_c100",
    ]:
        print(f"Running results for dataset: {dataset_name}")
        for method_name in [
            "uncalibrated",
            "dir_ODIR_pt",
            "ms_ODIR_pt",
            "vecS",
            # "iop_diag",
            "tempS",
            "dec2TS",
            "knnTS_kl_cv_ll",
            "knnTS_euc_cv_ll",
            "kernelTS_RBF_cv_ll",
            "kernelTS_DIR_cv_ll",
            "rfTS_cv_ll",
            "rf_cv_ll",
            "knn_kl_cv_ll",
            "kernel_DIR_cv_ll",
        ]:
            print(f"Running results for method: {method_name}")
            start = time.time()
            run(dataset_name=dataset_name, method_name=method_name)
            print(f"Time taken: {time.time() - start}")
            print()
