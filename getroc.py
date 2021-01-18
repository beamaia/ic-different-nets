import numpy as np
import matplotlib.pyplot as plt  
from sklearn import datasets, metrics, model_selection, svm

def get_info():
    # model_name = input("Model name:\nShould be 'resnet50', 'mobilenetv2' or 'inceptionv3'\n")
    model = "resnet50"
    date = input("Date:\n")
    version = input("Version:\n")
    return model, date, version

if __name__ == "__main__":
    model, date, version = get_info()

    PATH =  str(model) + "_" + str(date) + "_" + str(version) + ".txt"
    PATH_y = "./y_predict/y_test_" + PATH
    PATH_scores = "./y_predict/y_predict_" + PATH

    y = np.loadtxt(PATH_y)
    scores = np.loadtxt(PATH_scores)

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight(class_weight={0: 89, 1: 439}, y=y)
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
    print(precision, recall, thresholds, sep="\n")
    f1 = metrics.f1_score(y, scores)
    print(f"F1: {f1:.3f}")
    plt.plot(recall, precision, marker='.')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.ylim(0, 1)
    # show the plot
    plt.show()