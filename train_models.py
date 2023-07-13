# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from custom_struct import custom_ensemble, custom_stack_ensemble, custom_auto_encoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC


# %% Using raw data

train_test_splits = pd.concat(pd.read_excel("train_test_patients.xlsx", sheet_name=None)).reset_index(drop=True)

data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  
    
sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]
sample_patient_side = pd.DataFrame([[data_samples[g].index.get_level_values(0)[0],\
                                     data_samples[g].index.get_level_values(2)[0]]\
                                     for g in range(0,len(data_samples))],\
                                     columns=["Patient","Side"])

for k in range(0,5):
    train_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                 index[train_test_splits.iloc[:,3+k]=="TRAIN"],[0,1]],how="right")["index"].values
    test_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                index[train_test_splits.iloc[:,3+k]=="TEST"],[0,1]],how="right")["index"].values

    X_train = [data_samples[ti] for ti in train_idxs]
    y_train = [sample_classes[ti] for ti in train_idxs]
    X_test = [data_samples[ti] for ti in test_idxs]
    y_test = [sample_classes[ti] for ti in test_idxs]
    
    model_measures = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
    
    stack_model = SVC(probability=True)
    base_models = [SVC(probability=True) for _ in range(0, model_measures.shape[0])]
    base_models_preds = [[]] * model_measures.shape[0]
    
    for m in range(0, model_measures.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])

        base_models[m].fit(X_train_m, y_train)
        base_models_preds[m] = base_models[m].predict_proba(X_train_m)
        
    base_models_preds = np.column_stack(base_models_preds) 
    stack_model.fit(base_models_preds, y_train)
        
    base_models_preds = [[]] * model_measures.shape[0]
    
    for m in range(0, model_measures.shape[0]):
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        base_models_preds[m] = base_models[m].predict_proba(X_test_m)
        
    base_models_preds = np.column_stack(base_models_preds)
    y_pred = stack_model.predict(base_models_preds)
        
    model_report = pd.DataFrame(np.column_stack((np.array(y_test),y_pred)),columns=["y_true","y_pred"])
    
# %% Using autoencoders

train_test_splits = pd.concat(pd.read_excel("train_test_patients.xlsx", sheet_name=None)).reset_index(drop=True)

data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  
    
sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]
sample_patient_side = pd.DataFrame([[data_samples[g].index.get_level_values(0)[0],\
                                     data_samples[g].index.get_level_values(2)[0]]\
                                     for g in range(0,len(data_samples))],\
                                     columns=["Patient","Side"])

for k in range(0,5):
    train_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                 index[train_test_splits.iloc[:,3+k]=="TRAIN"],[0,1]],how="right")["index"].values
    test_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                index[train_test_splits.iloc[:,3+k]=="TEST"],[0,1]],how="right")["index"].values

    X_train = [data_samples[ti] for ti in train_idxs]
    y_train = [sample_classes[ti] for ti in train_idxs]
    X_test = [data_samples[ti] for ti in test_idxs]
    y_test = [sample_classes[ti] for ti in test_idxs]
    
    model_measures = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
    
    stack_model = SVC(probability=True)
    base_models = [SVC(probability=True) for _ in range(0, model_measures.shape[0])]
    base_models_preds = [[]] * model_measures.shape[0]     
    
    for m in range(0, model_measures.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        encoder_m = load_model("Autoencoders/E_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_train_m = encoder_m.predict(X_train_m)
        base_models[m].fit(X_train_m, y_train)
        base_models_preds[m] = base_models[m].predict_proba(X_train_m)
        
    base_models_preds = np.column_stack(base_models_preds) 
    stack_model.fit(base_models_preds, y_train)
    
    base_models_preds = [[]] * model_measures.shape[0]
    
    for m in range(0, model_measures.shape[0]):
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        encoder_m = load_model("Autoencoders/E_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_test_m = encoder_m.predict(X_test_m)
        base_models_preds[m] = base_models[m].predict_proba(X_test_m)
        
    base_models_preds = np.column_stack(base_models_preds)
    y_pred = stack_model.predict(base_models_preds)
        
    model_report = pd.DataFrame(np.column_stack((np.array(y_test),y_pred)),columns=["y_true","y_pred"])
    
# %% Using anomaly detection
        
train_test_splits = pd.concat(pd.read_excel("train_test_patients.xlsx", sheet_name=None)).reset_index(drop=True)

data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  
    
sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]
sample_patient_side = pd.DataFrame([[data_samples[g].index.get_level_values(0)[0],\
                                     data_samples[g].index.get_level_values(2)[0]]\
                                     for g in range(0,len(data_samples))],\
                                     columns=["Patient","Side"])        
    
for k in range(0,5):
    train_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                 index[train_test_splits.iloc[:,3+k]=="TRAIN"],[0,1]],how="right")["index"].values
    test_idxs = sample_patient_side.reset_index().merge(train_test_splits.iloc[train_test_splits.\
                index[train_test_splits.iloc[:,3+k]=="TEST"],[0,1]],how="right")["index"].values

    X_train = [data_samples[ti] for ti in train_idxs]
    y_train = [sample_classes[ti] for ti in train_idxs]
    X_test = [data_samples[ti] for ti in test_idxs]
    y_test = [sample_classes[ti] for ti in test_idxs]
    
    model_measures = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
    
    anomaly_detection_model = SVC(probability=True)
    mses = [[]] * model_measures.shape[0] 
    
    for m in range(0, model_measures.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        autoencoder_m = load_model("Anomaly Detection Autoencoders/AE_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_pred_m = autoencoder_m.predict(X_train_m)
        mses[m] = np.array([mean_squared_error(X_train_m[n,:],X_pred_m[n,:]) for n in range(0,X_train_m.shape[0])])
    
    mses = np.column_stack(mses) 
    
    y_train_anomaly_detection = np.array(y_train)
    y_train_anomaly_detection[y_train_anomaly_detection!="Normal"] = "Not Normal"
    anomaly_detection_model.fit(mses, y_train_anomaly_detection)
    
    stack_model = SVC(probability=True)
    base_models = [SVC(probability=True) for _ in range(0, model_measures.shape[0])]
    base_models_preds = [[]] * model_measures.shape[0]     
    y_train_not_normal = np.array(y_train)
    
    for m in range(0, model_measures.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        X_train_m = X_train_m[y_train_not_normal!="Normal"]
        encoder_m = load_model("Anomaly Detection Autoencoders/E_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_train_m = encoder_m.predict(X_train_m)
        base_models[m].fit(X_train_m, y_train_not_normal[y_train_not_normal!="Normal"])
        base_models_preds[m] = base_models[m].predict_proba(X_train_m)
        
    base_models_preds = np.column_stack(base_models_preds) 
    stack_model.fit(base_models_preds, y_train_not_normal[y_train_not_normal!="Normal"])
    
    # Test model    
    mses = [[]] * model_measures.shape[0] 
    
    for m in range(0, model_measures.shape[0]):
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        autoencoder_m = load_model("Anomaly Detection Autoencoders/AE_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_pred_m = autoencoder_m.predict(X_test_m)
        mses[m] = np.array([mean_squared_error(X_test_m[n,:],X_pred_m[n,:]) for n in range(0,X_test_m.shape[0])])
        
    mses = np.column_stack(mses) 
    y_pred = anomaly_detection_model.predict(mses)
    
    base_models_preds = [[]] * model_measures.shape[0]
    for m in range(0, model_measures.shape[0]):
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        X_test_m = X_test_m[y_pred!="Normal"]
        encoder_m = load_model("Anomaly Detection Autoencoders/E_{}_{}_{}_{}.keras".format(model_measures.iloc[m,0],model_measures.iloc[m,1],model_measures.iloc[m,2],str(k+1)))
        X_test_m = encoder_m.predict(X_test_m)
        base_models_preds[m] = base_models[m].predict_proba(X_test_m)
        
    base_models_preds = np.column_stack(base_models_preds)
    y_pred_not_normal = stack_model.predict(base_models_preds)
    
    y_pred[y_pred!="Normal"] = y_pred_not_normal
        
    model_report = pd.DataFrame(np.column_stack((np.array(y_test),y_pred)),columns=["y_true","y_pred"])