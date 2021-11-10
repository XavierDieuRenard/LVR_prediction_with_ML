# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:30:42 2020

@author: Xavier Dieu

===============================================================================
FEATURE SELECTION FUNCTIONS AND CLASSES FOR EZML
===============================================================================


"""
# IMPORTS
import pandas as pd
import numpy as np
import os
from datetime import datetime
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ezml.model_maker import make_DNNmodel


# FUNCTIONS

# None so far

# CLASSES

class feat_selector(BaseEstimator, TransformerMixin) :

    """
    custom transformer for ranking and selecting features through a LGBM or a DNN
    """
    # defaults class attributes params for lgbm classifier
    
    def __init__(self,  model_type='lgbm', mode='class', seed=0, outdir=None) :
        self.mode = mode
        self.seed = seed
        self.outdir = outdir
        self.support_ = None
        self.result_df = None
        self.poly_feat = None
        self.model_type = model_type
        if model_type == 'lgbm' :
            self.defaults = {'num_leaves': 20, 'max_depth': -1, 'class_weight':'balanced',
                'min_child_samples': 3}
        if model_type == 'dnn' :
            self.defaults = None

        
        
    def fit(self, X, y, feat_names, model_params=None, k=2, sample_weight=None, es=5,
            bootstrap = 5, test_size=0.2, poly = False, interaction_only = True) :
 
        if model_params is None :
            model_params = self.defaults
        
        # preprocessing with Polynomial features to add possible features extraction
        if poly :
            Poly_feat = PolynomialFeatures(degree = 2, interaction_only = interaction_only, include_bias = False)
            X_p = Poly_feat.fit_transform(X)
            feat_names = Poly_feat.get_feature_names(feat_names)
            X_p = pd.DataFrame(data=X_p, columns=feat_names)
            self.poly_feat = Poly_feat
        else :
            X_p = X
            
        # adding all combinations of k features 
        comb_feats = list(combinations(feat_names, k))
        feat_names.extend(comb_feats)
        
        # instantiating a dataframe to store results
        if self.mode == 'class' :
            score_name_train = 'AUC_train'
            score_name_val = 'AUC_val'
        elif self.mode == 'reg' :
            score_name_train = 'RMSE_train'
            score_name_val = 'RMSE_val'
        else :
            raise ValueError
            print('non valid mode argument')
        self.result_df = pd.DataFrame(columns=[score_name_train, score_name_val], index=feat_names)
        
        # fitting lgbm on each features or combination of features
        for feat in self.result_df.index : 

            X_tmp = X_p.loc[:,feat]
            
            score_train = list()
            score_val = list()

            for i in range(bootstrap) :                
                # split between train and validation data
                split = StratifiedShuffleSplit(n_splits=1, 
                                               test_size=test_size, 
                                               random_state=i)
                
                for train_index, val_index in split.split(X_tmp, y):
                    X_train = X_tmp.loc[train_index]
                    X_val = X_tmp.loc[val_index]
                    y_train = y.loc[train_index]
                    y_val = y.loc[val_index]
                
                X_train.reset_index(drop=True, inplace=True)
                X_val.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_val.reset_index(drop=True, inplace=True)

                if isinstance(feat, str) :
                    X_train = X_train.values.reshape(-1, 1)
                    X_val = X_val.values.reshape(-1, 1)
                                
                if self.mode == 'class' :
                    
                    if self.model_type == 'lgbm' :

                        final_model = LGBMClassifier(random_state=self.seed).set_params(**model_params)
        
                        final_model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=(X_val, y_val),
                                 early_stopping_rounds=es)            
                        
                    if self.model_type == 'dnn' :
                        
                        if model_params is None :
                            model_params = dict(input_dim = X_train.shape[1],
                                                batch_size=32, 
                                                epochs=300,
                                                callbacks = [EarlyStopping(monitor= 'val_loss', patience=10, verbose=1),
                                                             ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)],
                                                kernel_initializer = 'lecun_normal',
                                                kernel_seed = self.seed,
                                                activation = 'selu',
                                                dropout_rate = None,
                                                n_output = 1,
                                                activation_output = 'sigmoid',
                                                loss = 'binary_crossentropy',
                                                optimizer = 'nadam',
                                                lr = 0.001,
                                                layers = [k*10,k*10],
                                                weight_l2 = None,
                                                weight_l2_output = None,
                                                batch_norm = None, 
                                                validation_data=(X_val, y_val)
                                                )

                        final_model = KerasClassifier(make_DNNmodel, 
                                                      **model_params)
        
                        final_model.fit(X_train, y_train)            
                        
                    y_val_predict_p = final_model.predict_proba(X_val)
                    if y_val_predict_p.shape[1] == 2 :
                        clf_auc = roc_auc_score(y_val, y_val_predict_p[:,1])
                    else :
                        clf_auc = roc_auc_score(y_val, y_val_predict_p)
                    
                    score_val.append(clf_auc)
                    
                    y_train_predict_p = final_model.predict_proba(X_train)                      
                    if y_train_predict_p.shape[1] == 2 :
                        clf_auc_train = roc_auc_score(y_train, y_train_predict_p[:,1])
                    else :
                        clf_auc_train = roc_auc_score(y_train, y_train_predict_p)
 
                    score_train.append(clf_auc_train)                    

            self.result_df.loc[feat, score_name_train] = np.median(score_train)
            self.result_df.loc[feat, score_name_val] = np.median(score_val)
        
        # exports of results 
        if self.outdir != None :
            now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            outpath = os.path.join(self.outdir,'CART_selector_{}.xlsx'.format(now))
            self.result_df.to_excel(outpath)  
    
        return self.result_df
    
    def transform(self, X, feat_names, threshold=20, feat_list = None) :
        
        if feat_list == None :
            if self.mode == 'class' :
                self.result_df = self.result_df.sort_values('AUC_val', ascending = False)
            self.result_df = self.result_df.head(threshold)
        
        self.support_ = list()
        if feat_list != None :
            feat_to_add = feat_list
        else :
            feat_to_add = list(self.result_df.index)
                
        for feat in feat_to_add :
            if isinstance(feat, str) :            
                if feat in self.support_ :
                    pass
                else :
                    self.support_.append(feat)
            if isinstance(feat, tuple) :            
                for tup_feat in feat :
                    if tup_feat in self.support_ :
                        pass
                    else :
                        self.support_.append(tup_feat)
                
        if self.poly_feat != None :
            X_p = self.poly_feat.fit_transform(X)
            feat_names = self.poly_feat.get_feature_names(feat_names)
            X_p = pd.DataFrame(data=X_p, columns=feat_names)
        else :
            X_p = X
        
        return X_p.loc[:, self.support_]




