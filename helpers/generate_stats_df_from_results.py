from .load_dataset_method_combination import load_dataset, load_result
from .multiclass_CE import classwise_equal_size_ece, confidence_equal_size_ece
from .optimal_decisions import generate_normal_cost_matrices, true_loss_predicted_loss
import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

def rename_df(df):
    return df.rename(index={"bs":'Brier score', "ll":"log-loss", "cw_ece":"classwise ECE", "conf_ece":"confidence ECE",
                 "avg_loss_gap2":"loss gap",
                            "resnet_wide32": "ResNet Wide 32",
                            "densenet40":"DenseNet-40",
                            "resnet110":"ResNet-110",
                            "c10":"C-10", "c100":"C-100"},
          columns={"kernel_DIR_cv_ll":"kernel$_{DIR}$",
                   "knn_euc_cv_ll":"KNN$_{euc}$",
                   "rf_cv_ll":"RF",
                   "tempS":"TS",
                   "dir_ODIR_p":"DIR",
                   "ms_ODIR_p":"MS",
                    "vecS":"VS",
                   "iop_diag":"IOP",
                   "action=2_sum1":"DEC",
                   "knnTS_kl_cv_ll": "KNN",
                   #"knnIOP_kl_cv_ll": "IOP KNN_{KL}",
                   "uncalibrated": "uncal",
                   "knn_kl_cv_ll": "KNN$_{kl}$"
                   })

def add_ranks_to_df(df_in, ascending=True):
    df = pd.DataFrame.copy(df_in, deep=True)
    ranks = df.rank(axis=1, ascending=ascending, method="min")
    # Combine ranks and data
    for row_idx in range(len(df)):
        for column_idx in range(len(df.iloc[row_idx])):

            item = df.iloc[row_idx, column_idx]
            rank = ranks.iloc[row_idx, column_idx]

            item = np.round(item, 7)
            df.iloc[row_idx, column_idx] = str(item) + "_{" + str(int(rank)) + "}"
            if rank == 1:
                df.iloc[row_idx, column_idx] = "\mathbf{" + df.iloc[row_idx, column_idx] + "}"
            df.iloc[row_idx, column_idx] = "$" + df.iloc[row_idx, column_idx] + "$"
    return df


def df_to_latex(df):
    with pd.option_context("max_colwidth", 25):
        output = df.to_latex(escape=False)
        return output


def add_double_column_header(latex, header_names, header_widths):
    separator_ids = np.cumsum([8, 2])[:-1]
    lines = latex.splitlines()
    output = ""
    for i in range(len(lines)):
        if i != 2:
            output += lines[i] + "\n"
        else:
            for header_idx in range(len(header_names)):
                if header_idx != 0:
                    output += "&"
                output += "\multicolumn{" + str(header_widths[header_idx]) +"}"
                if header_idx != 0:
                    output += "{c}{" # "{|c}{" #
                else:
                    output += "{c}{"
                output += header_names[header_idx] +"}"

            output += "\\\\\n"
            col_names = lines[2].replace(" ", "").replace("\\", "").split("&")
            for col_idx in range(len(col_names)):
                if col_idx != 0:
                    output += "&"
                output += "\multicolumn{1}{"
                if col_idx in separator_ids:
                    output += ""# "|"
                output += "c}{" + col_names[col_idx] + "}"
            output += "\\\\"

    return output


def add_hline(latex, ids, lengths):
    lines = latex.splitlines()
    output = ""
    id_idx = 0
    for i in range(len(lines)):
        output += lines[i] + "\n"
        if i in ids:
            output += "\cmidrule{" + str(lengths[id_idx][0]) + "-" + str(lengths[id_idx][1]) +"}\n"
            id_idx += 1
    return output


def create_table(datasets, metric, methods, add_avg_rank=True, rounding=3, n_matrices = 500):
    df = {}

    for (dataset, model) in datasets:
        dataset_model = model + "_" + dataset
        print(dataset_model)
        df[(dataset, model)] = {}

        logits_train, p_train, y_train, y_train_flat, logits_test, p_test, y_test, y_test_flat = load_dataset(dataset_model)

        for cal_method in methods:
            try:
                precomputed_results = load_result(dataset_model, cal_method)
                cal_p_test = precomputed_results["cal_p_test"]
                #print(precomputed_results["cv_param"])
            except:
                cal_p_test = np.ones(p_test.shape)

            if metric == "bs":
                df[(dataset, model)][cal_method] = np.round(np.mean(np.sum((cal_p_test - y_test) ** 2, axis=1)), rounding)
            elif metric == "ll":
                df[(dataset, model)][cal_method] = np.round(log_loss(y_test, cal_p_test), 3)
            elif metric == "cw_ece":
                df[(dataset, model)][cal_method] = np.round(100 * classwise_equal_size_ece(p=cal_p_test, y=y_test, n_bins=15, threshold=0)["ece_abs"], rounding)
            elif metric == "conf_ece":
                df[(dataset, model)][cal_method] = np.round(100 * confidence_equal_size_ece(p=cal_p_test, y=y_test, n_bins=15)["ece_abs"], rounding)
            elif metric == "accuracy":
                df[(dataset, model)][cal_method] = np.round(np.mean(y_test.argmax(axis=1) == cal_p_test.argmax(axis=1)), rounding)
            elif metric == "avg_loss2":
                cost_matrices2 = generate_normal_cost_matrices(n_classes=y_test.shape[1], n_decisions=2, n_matrices=n_matrices, seed=0, mean=0, std=1)
                true_loss, pred_loss = true_loss_predicted_loss(cal_p_test, y_test, cost_matrices2)
                df[(dataset, model)][cal_method] = np.round(np.mean(true_loss), rounding)
            elif metric == "avg_loss_gap2":
                cost_matrices2 = generate_normal_cost_matrices(n_classes=y_test.shape[1], n_decisions=2, n_matrices=n_matrices, seed=0, mean=0, std=1)
                true_loss, pred_loss = true_loss_predicted_loss(cal_p_test, y_test, cost_matrices2)
                df[(dataset, model)][cal_method] = np.round(np.mean(np.mean(np.abs(true_loss - pred_loss))), rounding)

    df = pd.DataFrame.from_dict(df).T
    if metric == "accuracy":
        dfn = add_ranks_to_df(df, ascending=False)
    else:
        dfn = add_ranks_to_df(df, ascending=True)
    if add_avg_rank:
        if metric == "accuracy":
            ranks = df.rank(axis=1, ascending=False, method="min")
        else:
            ranks = df.rank(axis=1, method="min")

        dfn.loc[("", "average rank"),:] = np.round(np.mean(ranks), 1)
    return dfn