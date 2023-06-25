# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from custom_struct import custom_ensemble, custom_stack_ensemble, custom_auto_encoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# %%
data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  
    
#data_agg_legs = data_agg_legs[data_agg_legs.index.get_level_values(6)!="Normal"]

sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]

skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(np.zeros(len(sample_classes)),np.array(sample_classes))

for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(sample_classes)),np.array(sample_classes))):
    X_train = [data_samples[ti] for ti in train_index]
    y_train = [sample_classes[ti] for ti in train_index]
    X_test = [data_samples[ti] for ti in test_index]
    y_test = [sample_classes[ti] for ti in test_index]
    
    model = custom_auto_encoder()
    model.fit(X_train, X_test, y_train, y_test)




# %% Voting Ensemble
data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  

sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]

skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(np.zeros(len(sample_classes)),np.array(sample_classes))

for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(sample_classes)),np.array(sample_classes))):
    X_train = [data_samples[ti] for ti in train_index]
    y_train = [sample_classes[ti] for ti in train_index]
    X_test = [data_samples[ti] for ti in test_index]
    y_test = [sample_classes[ti] for ti in test_index]
    
    model = custom_ensemble()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mtx_disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    conf_mtx_disp.plot()
    plt.show()

# %% Stack Ensemble
data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  
    
#data_agg_legs = data_agg_legs[data_agg_legs.index.get_level_values(6)!="Normal"]

sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]

skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(np.zeros(len(sample_classes)),np.array(sample_classes))

for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(sample_classes)),np.array(sample_classes))):
    X_train = [data_samples[ti] for ti in train_index]
    y_train = [sample_classes[ti] for ti in train_index]
    X_test = [data_samples[ti] for ti in test_index]
    y_test = [sample_classes[ti] for ti in test_index]
    
    model = custom_stack_ensemble()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mtx_disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    conf_mtx_disp.plot()
    plt.show()