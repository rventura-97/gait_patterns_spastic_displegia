# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, precision_score, recall_score

# %%
report = np.zeros((12,13), dtype='object')

row = 0
for appr in ["Raw", "AE", "Anomaly"]:
    for algo in ["ANN","KNN","LR","SVM"]:
        outputs = [[]] * 5
        metrics = np.zeros((5,11))
        for f in range(0, len(outputs)):
            outputs[f] = pd.read_csv("Outputs/{}/{}/{}.csv".format(appr,algo,str(f+1)))
            metrics[f,0] = accuracy_score(outputs[f].y_true, outputs[f].y_pred)
            metrics[f,1] = cohen_kappa_score(outputs[f].y_true, outputs[f].y_pred)
            metrics[f,2] = matthews_corrcoef(outputs[f].y_true, outputs[f].y_pred)
            metrics[f,3:7] = precision_score(outputs[f].y_true, outputs[f].y_pred,average=None,labels=["Apparent Equinus","Crouch Gait","Jump Gait","True Equinus"])
            metrics[f,7:] = recall_score(outputs[f].y_true, outputs[f].y_pred,average=None,labels=["Apparent Equinus","Crouch Gait","Jump Gait","True Equinus"])
        metrics_mean = np.mean(metrics, axis=0)
        metrics_std = np.std(metrics, axis=0)
        
        report[row,0] = appr
        report[row,1] = algo
        report[row,2:] = [str(np.round(metrics_mean[i],2)) + " ± " + str(np.round(metrics_std[i],2)) for i in range(0,metrics_mean.size)]
        row += 1
        
report = pd.DataFrame(data=report,\
                      columns=["Approach","Algorithm","Accuracy","Kappa",\
                               "Matthews","Apparent Equinus","Crouch Gait",\
                               "Jump Gait","True Equinus","Apparent Equinus","Crouch Gait",\
                               "Jump Gait","True Equinus"])

report.to_excel("REPORT.xlsx")