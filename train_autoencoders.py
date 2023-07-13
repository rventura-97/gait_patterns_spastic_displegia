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

# %% Train autoencoders

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
    
    autoencoders_vars = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
    
    for m in range(0, autoencoders_vars.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        
        ros = RandomOverSampler()
        X_train_m, y_train_m = ros.fit_resample(X_train_m, y_train)
        X_test_m, y_test_m = ros.fit_resample(X_test_m, y_test)
    
        n_inputs = X_train_m.shape[1]

        # define encoder
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = 10
        bottleneck = Dense(n_bottleneck)(e)


        # define decoder, level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_train_m, X_train_m, epochs=200, batch_size=16, verbose=2, validation_data=(X_test_m, X_test_m))
        
        model = Model(inputs=visible, outputs=bottleneck)
        model.save("Autoencoders/E_{}_{}_{}_{}.keras".format(autoencoders_vars.iloc[m,0],autoencoders_vars.iloc[m,1],autoencoders_vars.iloc[m,2],str(k+1)))
    

# Train autoencoders for anomaly detection architecture
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
    
    autoencoders_vars = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
    
    for m in range(0, autoencoders_vars.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        X_train_m = X_train_m[np.array(y_train) == "Normal",:]
        
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        X_test_m = X_test_m[np.array(y_test) == "Normal",:]
        
    
        n_inputs = X_train_m.shape[1]

        # define encoder
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = 10
        bottleneck = Dense(n_bottleneck)(e)


        # define decoder, level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_train_m, X_train_m, epochs=200, batch_size=16, verbose=2, validation_data=(X_test_m, X_test_m))
        
        model.save("Anomaly Detection Autoencoders/AE_{}_{}_{}_{}.keras".format(autoencoders_vars.iloc[m,0],autoencoders_vars.iloc[m,1],autoencoders_vars.iloc[m,2],str(k+1)))
        
    for m in range(0, autoencoders_vars.shape[0]):
        X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
        X_train_m = X_train_m[np.array(y_train) != "Normal",:]
        X_train_m = np.tile(X_train_m,(3,1))
        
        X_test_m = np.array([X_test[i].iloc[m,:].values for i in range(0,len(X_test))])
        X_test_m = X_test_m[np.array(y_test) != "Normal",:]
        X_test_m = np.tile(X_test_m,(3,1))
        
    
        n_inputs = X_train_m.shape[1]

        # define encoder
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = 10
        bottleneck = Dense(n_bottleneck)(e)


        # define decoder, level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_train_m, X_train_m, epochs=200, batch_size=16, verbose=2, validation_data=(X_test_m, X_test_m))
        
        model = Model(inputs=visible, outputs=bottleneck)
        model.save("Anomaly Detection Autoencoders/E_{}_{}_{}_{}.keras".format(autoencoders_vars.iloc[m,0],autoencoders_vars.iloc[m,1],autoencoders_vars.iloc[m,2],str(k+1)))
        
        

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
    
    y_pred = model.predict(X_test)
    conf_mtx_disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    conf_mtx_disp.plot()
    plt.show()



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