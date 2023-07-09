# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import numpy as np
from collections import Counter
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler

class custom_ensemble:
    
    def __init__(self):
        self.models = []
        self.models_measures = []
        
    def fit(self, X, y):
        self.models_measures = X[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
        self.models = [SVC() for _ in range(0, self.models_measures.shape[0])]
        
        for m in range(0, self.models_measures.shape[0]):
            X_m = np.array([X[i].iloc[m,:].values for i in range(0,len(X))])
            
            self.models[m].fit(X_m, y)
            
        print("")
            
    def predict(self, X):
        y_pred = [''] * len(X)
        base_models_preds = [[]] * self.models_measures.shape[0]
        
        for m in range(0, self.models_measures.shape[0]):
            X_m = np.array([X[i].iloc[m,:].values for i in range(0,len(X))])
            base_models_preds[m] = self.models[m].predict(X_m)
            
        base_models_preds = np.array(base_models_preds)  
        
        for k in range(0, len(X)):
            class_votes = Counter(base_models_preds[:,k])
            y_pred[k] = max(class_votes)
            
            
        return y_pred
        
                               
class custom_stack_ensemble:
    
    def __init__(self):
        self.models = []
        self.models_measures = []
        self.stack_model = []
        
    def fit(self, X, y):
        self.models_measures = X[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
        self.models = [SVC(probability=True, verbose=False) for _ in range(0, self.models_measures.shape[0])]
        
        base_models_preds = [[]] * self.models_measures.shape[0]
        
        for m in range(0, self.models_measures.shape[0]):
            X_m = np.array([X[i].iloc[m,:].values for i in range(0,len(X))])
            self.models[m].fit(X_m, y)
            base_models_preds[m] = self.models[m].predict_proba(X_m)
            
        base_models_preds = np.column_stack(base_models_preds)
        
        self.stack_model = SVC(probability=True, verbose=False)
        self.stack_model.fit(base_models_preds, y)
 
            
    def predict(self, X):
        
        base_models_preds = [[]] * self.models_measures.shape[0]
        
        for m in range(0, self.models_measures.shape[0]):
            X_m = np.array([X[i].iloc[m,:].values for i in range(0,len(X))])
            base_models_preds[m] = self.models[m].predict_proba(X_m)
            
        base_models_preds = np.column_stack(base_models_preds)
        y_pred = self.stack_model.predict(base_models_preds)
            
        return y_pred                               


class custom_auto_encoder:
    def __init__(self):
        self.autoencoders = []
        self.encoders = []
        self.base_models = []
        self.stack_model = []
        self.models_measures = []
        
        
    def fit(self, X_train, X_valid, y_train, y_valid):
        
        self.models_measures = X_train[0].index.to_frame().iloc[:,[3,4,5]].reset_index(drop=True)
        
        self.autoencoders = [[]] * self.models_measures.shape[0]
        self.encoders = [[]] * self.models_measures.shape[0]
        
        for m in range(0, self.models_measures.shape[0]):
            X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
            X_valid_m = np.array([X_valid[i].iloc[m,:].values for i in range(0,len(X_valid))])
            
            ros = RandomOverSampler()
            X_train_m, y_train_m = ros.fit_resample(X_train_m, y_train)
            X_valid_m, y_valid_m = ros.fit_resample(X_valid_m, y_valid)
        
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
            
            history = model.fit(X_train_m, X_train_m, epochs=200, batch_size=16, verbose=2, validation_data=(X_valid_m, X_valid_m))
            
            # plt.plot(history.history['loss'], label='train')
            # plt.plot(history.history['val_loss'], label='test')
            # plt.legend()
            # plt.show()
            # define an encoder model (without the decoder)
            self.autoencoders[m] = model
            self.encoders[m] = Model(inputs=visible, outputs=bottleneck)

            # save the encoder to file
            #encoder.save('encoder.h5')
        self.base_models = [SVC(probability=True, verbose=False) for _ in range(0, self.models_measures.shape[0])]    
        base_models_preds = [[]] * self.models_measures.shape[0]
        for m in range(0, self.models_measures.shape[0]):
            X_train_m = np.array([X_train[i].iloc[m,:].values for i in range(0,len(X_train))])
            X_train_m = self.encoders[m].predict(X_train_m)
            self.base_models[m].fit(X_train_m, y_train)
            base_models_preds[m] = self.base_models[m].predict_proba(X_train_m)
            
        base_models_preds = np.column_stack(base_models_preds)
        
        self.stack_model = SVC(probability=True, verbose=False)
        self.stack_model.fit(base_models_preds, y_train)

            
    def predict(self, X):
        
        base_models_preds = [[]] * self.models_measures.shape[0]
        
        for m in range(0, self.models_measures.shape[0]):
            X_m = np.array([X[i].iloc[m,:].values for i in range(0,len(X))])
            X_m = self.encoders[m].predict(X_m)
            base_models_preds[m] = self.base_models[m].predict_proba(X_m)
            
        base_models_preds = np.column_stack(base_models_preds)
        y_pred = self.stack_model.predict(base_models_preds)
            
        return y_pred 
        
class custom_anomaly_classifier:
    def __init__(self):
        pass
    
    
    