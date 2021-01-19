import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd

from sklearn import metrics, svm
from sklearn.utils.class_weight import compute_sample_weight

import sys

from glob import glob
import fnmatch

# def get_info():
#     # model_name = input("Model name:\nShould be 'resnet50', 'mobilenetv2' or 'inceptionv3'\n")
#     model = "resnet50"
#     date = input("Date:\n")
#     version = input("Version:\n")
#     sets = int(input("Set used:\n(1) First Set (2) Second Set (3) Both Sets\n"))
#
#     if sets is 1:
#         positive_class = 1978
#         negative_class = 398
#     elif sets is 2:
#         positive_class = 2405
#         negative_class = 889
#     elif sets is 3:
#         positive_class = 1311
#         negative_class = 4197
#     else:
#         sys.exit(-1)
#
#     return model, date, version, positive_class, negative_class

def get_paths():
    PATH_y_pred = "./y_predict/y_predict_*"
    PATH_y_true = "./y_predict/y_test_*"

    pred = glob(PATH_y_pred)
    true = glob(PATH_y_true)

    if len(pred) != len(true):
        print("y_predict is different than y_true!")
        sys.exit(-1)

    return pred, true

def get_info(pred_path, true_path):
    _, _, pred_string = pred_path.split("/")
    _, _, true_string = true_path.split("/")
    pred_string, _ = pred_string.split(".")
    true_string, _ = true_string.split(".")
    _, pred_string = pred_string.split("y_predict_")
    _, true_string = true_string.split("y_test_")
    pred_model, pred_date, pred_version = pred_string.split("_")
    true_model, true_date, true_version = true_string.split("_")
    
    same_model = pred_model == true_model
    same_date = pred_date == true_date
    same_version = pred_version == true_version

    if same_model and same_version and same_date:
        return pred_model, pred_date, pred_version
    else: 
        print("Paths are for different models!")
        sys.exit(-1)


def get_accuracy(df, model, date, version):
    same_everything = df.loc[(df["Model"]==model) & (df["Date"] == date) & (df["Version"] == int(version))]
    index_size = len(same_everything.index)

    if index_size is not 1:
        print(f"Got {index_size} instead of 1")
        sys.exit(-1)
    else:
        return same_everything["Accuracy"].values[0], same_everything["Set"].values[0]


def get_stats():
    # model, date, version, positive_class, negative_class = get_info()
    df_accuracies = pd.read_csv("./test_accuracies/accuracies.csv")

    pred, true = get_paths()    
    pred.sort(), true.sort()
    
    models = []
    dates = []
    versions = []
    tn_list = []
    tp_list = []
    fn_list = []
    fp_list = []
    precisions = []
    recalls = []
    f1_scores = []
    # auc_scores = []
    accuracies = []
    sets = []

    for pred_path, true_path in zip(pred, true):
        model, date, version = get_info(pred_path, true_path)
        accuracy, set_used = get_accuracy(df_accuracies, model, date, version)

        models.append(model)
        dates.append(date)
        versions.append(version)
        accuracies.append(float(accuracy)/100)
        sets.append(set_used)

        y_pred = np.loadtxt(pred_path)
        y_true = np.loadtxt(true_path)

        # y_true1 = np.count_nonzero(y_true)
        # y_true0 = len(y_true) - y_true1
        # weights = compute_sample_weight({1:y_true1, 0: y_true0}, y_true)
  
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

        tn_list.append(tn)
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)

        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)

        # auc = metrics.roc_auc_score(y_true, y_pred, sample_weight=weights)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        # auc_scores.append(auc)

    tn_list = np.array(tn_list)
    tp_list = np.array(tp_list)
    fn_list = np.array(fn_list)
    fp_list = np.array(fp_list)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    # auc_scores = np.array(auc_scores)
    accuracies = np.array(accuracies)
    sets = np.array(sets)

    dict_stats = {"Model": models, 
             "Date": dates,
             "Version": versions,
             "Set": sets,
             "Accuracy": accuracies, 
             "Precision": precisions,
             "Recall": recalls,
             "F1": f1_scores}
            #  "AUC": auc_scores}

    dict_pos_neg = {"Model": models, 
             "Date": dates,
             "Version": versions,
             "Set": sets,
             "True Positive": tp_list,
             "False Negatives": fn_list,
             "False Positives": fp_list, 
             "True Negatives": tn_list}

    stats = pd.DataFrame(dict_stats)
    pos_neg = pd.DataFrame(dict_pos_neg)
    return stats, pos_neg

if __name__ == "__main__":
    stats, pos_neg = get_stats()
    stats.to_csv("stats.csv")
    pos_neg.to_csv("positive_negatives_stats.csv")
