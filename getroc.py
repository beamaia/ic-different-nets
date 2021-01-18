import numpy as np
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight

import sys

def get_info():
    # model_name = input("Model name:\nShould be 'resnet50', 'mobilenetv2' or 'inceptionv3'\n")
    model = "resnet50"
    date = input("Date:\n")
    version = input("Version:\n")
    sets = int(input("Set used:\n(1) First Set (2) Second Set (3) Both Sets\n"))

    if sets is 1:
        positive_class = 439
        negative_class = 89
    elif sets is 2:
        positive_class = 2405
        negative_class = 889
    elif sets is 3:
        positive_class = 1311
        negative_class = 4197
    else:
        sys.exit(-1)

    return model, date, version, positive_class, negative_class

if __name__ == "__main__":
    model, date, version, positive_class, negative_class = get_info()
    columns = ["Model name", "Date", "Version", "Set", "Precision", "Recall", "F1", "AUC"]
    
    PATH =  str(model) + "_" + str(date) + "_" + str(version) + ".txt"
    PATH_y = "./y_predict/y_test_" + PATH
    PATH_scores = "./y_predict/y_predict_" + PATH

    y = np.loadtxt(PATH_y)
    scores = np.loadtxt(PATH_scores)

    len_y = len(y)
    pos_y = np.sum(y)
    pos_scores = np.sum(scores)
    neg_y = len_y - pos_y
    neg_scores = len_y - pos_scores

    print(f"len: {len_y}")
    print(f"positive y: {pos_y} Negative y: {neg_y}")
    print(f"positive scores: {pos_scores} Negative scores: {neg_scores}")

    average_precision = metrics.average_precision_score(y, scores)
    print(f'Average precision-recall score: {average_precision}')

    sample_weight = compute_sample_weight(class_weight={0: negative_class, 1: positive_class}, y=y)
    print(f"Sample weight: {sample_weight}")

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1, sample_weight=sample_weight)
    roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # display.plot()  
    # plt.show() 
    print("FPR, TPR, Thresholds:")     
    print(fpr, tpr, thresholds, sep="\n")
    print(f'AUC: {roc_auc: .3f}')
    precision, recall, thresholds = metrics.precision_recall_curve(y, scores)
    print("Precision, recall, thresholds:")
    print((precision), (recall), thresholds, sep="\n")
    f1 = metrics.f1_score(y, scores)
    print(f"F1: {f1:.3f}")

                                                 
    # plt.plot(recall, precision, marker='.')
    # # axis labels
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim(0, 1)
    # # show the plot
    # plt.show()