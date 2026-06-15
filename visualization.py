import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# ROC
def roc(y_true, pred_dict, save_name):
    plt.figure()
    names = list(pred_dict.keys())
    preds = list(pred_dict.values())
    for i in range(len(names)):
        fpr, tpr, _ = roc_curve(y_true, preds[i])
        auc = roc_auc_score(y_true, preds[i])
        plt.plot(fpr, tpr, label=names[i] + ' (AUC=' + str(round(auc,3)) + ')')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(save_name)
    plt.close()
    print('ROC Save：', save_name)