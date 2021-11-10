# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:30:42 2020

@author: Xavier Dieu

===============================================================================
FUNCTIONS AND CLASSES TO SEARCH BEST MODELS FOR EZML
===============================================================================


"""

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV 
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc, roc_curve
from xgboost import XGBClassifier
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
#from imblearn.keras import BalancedBatchGenerator #deprecated
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC as keras_AUC
from ezml.model_maker import make_DNNmodel


# FUNCTIONS


# CLASSES

class SearchPipeline :
    
    """
    Easy Machine Learning pipeline class
    """
    
    
    def __init__(self, train_data, train_target, feats) :
        self.train_x = train_data
        self.train_y = train_target
        if isinstance(feats, pd.Series) :
            self.features = feats
        else :
             self.features = pd.Series(feats)           
        self.results_ = None
        self.DNNresults_ = None
    

    def search_model(self, 
                      models_grid,
                      outdir,
                      random_iter = range(3),
                      test_size = 0.25,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search_splits = 5,
                      grid_search_test_size = 0.25,
                      rfe = True,
                      plot_rfe = False,
                      rfe_step = 1,
                      min_rfe = 1,
                      rfe_model = None,
                      rfe_params = None,
                      n_jobs=-2,
                      preprocess_pipelines = {'passthrough':'passthrough'}, 
                      feature_selection_pipelines = {'no_feature_selection':'passthrough'}
                      ) :
        
        """
        Evaluating classifier pipelines through nested train/val split
        """
        
        # creating a df to store each pipeline results
        grid_cols = ['preprocessing', 
                 'feature_selection', 
                 'model', 
                 'hyperparameters',
                 'AUC_train',
                 'AUC',
                 'Accuracy',
                 'Balanced_Accuracy',
                 'Precision',
                 'Recall',
                 'hyperparameters with rfecv',
                 'AUC_train with rfecv',
                 'AUC with rfecv',    
                 'Accuracy with rfecv',
                 'Balanced_Accuracy with rfecv',
                 'Precision with rfecv',
                 'Recall with rfecv',
                 'num_seed',
                 'features']
        
        grid_rows = range(len(preprocess_pipelines)*len(feature_selection_pipelines)*len(models_grid)*len(random_iter))
        
        grid_df =  pd.DataFrame(columns=grid_cols, index=grid_rows)
        
                
        # setting a count for storing info in grid_df
        grid_df_row_count = 0
        
        # starting the first loop over the random_iter splits/random_state
        for i in random_iter :
            
            # split between train and validation data
            split = StratifiedShuffleSplit(n_splits=1, 
                                           test_size=test_size, 
                                           random_state=i)
            
            for train_index, val_index in split.split(self.train_x, self.train_y):
                X_train = self.train_x.loc[train_index]
                X_val = self.train_x.loc[val_index]
                y_train = self.train_y.loc[train_index]
                y_val = self.train_y.loc[val_index]

                        
            # Looping over each preprocessing pipeline submitted 
            for preprocess_pipeline in preprocess_pipelines.values() :

                selected_features = self.features.copy()

                if preprocess_pipeline == 'passthrough' :
                    X_p = X_train.values
                    X_val_p = X_val.values
                else :
                    X_p = preprocess_pipeline.fit_transform(X_train)
                    X_val_p = preprocess_pipeline.transform(X_val)    
    
                    try :
                        preprocess_pipeline.named_transformers_.num_pipe['polynomial']
                    except KeyError :
                        pass
                    else :
                        selected_features = preprocess_pipeline.named_transformers_.num_pipe['polynomial'].get_feature_names(selected_features)
                        selected_features = pd.Series(selected_features)
                        
                # Under sampling if not none :
                if under_sampling == 'random' :
                    X_p, y_train = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=i).fit_resample(X_p, y_train) 
                
                if under_sampling == 'ENN' :
                    X_p, y_train = EditedNearestNeighbours().fit_resample(X_p, y_train) 
                
                
                # Looping over each feature selection(s) :
                for feature_selection in feature_selection_pipelines.values() :
            
                    if feature_selection == 'passthrough' :
                        X_fs = X_p
                        X_val_fs = X_val_p
                    else : 
                        X_fs = feature_selection.fit_transform(X_p, y_train)
                        X_val_fs = feature_selection.transform(X_val_p)
                        selected_features  = selected_features[feature_selection.support_]
                
    
                # Looping over each models_grid dict (hyperparameters + data split variation)
                    for model in models_grid.keys() :
                                            
                        # grid search inner test/val split
                        clf = GridSearchCV(estimator=model, 
                                           param_grid=models_grid[model],
                                           scoring = 'roc_auc',
                                           cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                                     test_size=grid_search_test_size,
                                                                     random_state=i),
                                           n_jobs=n_jobs)
                        
                        clf.fit(X_fs, y_train)
                        
                        # computing metrics on val data
                        y_val_predict_p = clf.best_estimator_.predict_proba(X_val_fs)                      
                        if y_val_predict_p.shape[1] == 2 :
                            clf_auc = roc_auc_score(y_val, y_val_predict_p[:,1])
                        else :
                            clf_auc = roc_auc_score(y_val, y_val_predict_p)
                        
                        y_train_predict_p = clf.best_estimator_.predict_proba(X_fs)                      
                        if y_train_predict_p.shape[1] == 2 :
                            clf_auc_train = roc_auc_score(y_train, y_train_predict_p[:,1])
                        else :
                            clf_auc_train = roc_auc_score(y_train, y_train_predict_p)
                        
                        y_val_predict = clf.best_estimator_.predict(X_val_fs)
                        clf_acc = accuracy_score(y_val, y_val_predict)
                        clf_bal_acc = balanced_accuracy_score(y_val, y_val_predict)
                        clf_pre = precision_score(y_val, y_val_predict)
                        clf_rec = recall_score(y_val, y_val_predict)
                                                
                        
                        # writing pipeline results in grid_df
                        grid_df.loc[grid_df_row_count, 'preprocessing'] = \
                            list(preprocess_pipelines.keys())[list(preprocess_pipelines.values()).index(preprocess_pipeline)]
                       
                        grid_df.loc[grid_df_row_count, 'feature_selection'] = \
                            list(feature_selection_pipelines.keys())[list(feature_selection_pipelines.values()).index(feature_selection)]
                        
                        grid_df.loc[grid_df_row_count, 'model'] = \
                            str(model)[:re.search('\(', str(model)).span()[0]]
                        
                        grid_df.loc[grid_df_row_count, 'hyperparameters'] = \
                            str(clf.best_params_)
                        
                        grid_df.loc[grid_df_row_count, 'AUC_train'] = round(clf_auc_train, 3)
                        grid_df.loc[grid_df_row_count, 'AUC'] = round(clf_auc, 3)
                        grid_df.loc[grid_df_row_count, 'Accuracy'] = round(clf_acc, 3)
                        grid_df.loc[grid_df_row_count, 'Balanced_Accuracy'] = round(clf_bal_acc, 3)
                        grid_df.loc[grid_df_row_count, 'Precision'] = round(clf_pre, 3)
                        grid_df.loc[grid_df_row_count, 'Recall'] = round(clf_rec, 3)
                                
                        
                        # Add extra RFECV plus redo the grid search
                        # it will fine tune the number of features and remove collinearity
                        if rfe :
                            
                            if rfe_model == 'XGBoost' :
                                rfe_clf = XGBClassifier(n_jobs=-2, random_state=i).set_params(**rfe_params)
                            
                            elif rfe_model == 'eli5' :
                                rfe_clf = PermutationImportance(clf.best_estimator_, 
                                                                scoring='roc_auc',
                                                                cv=None, 
                                                                random_state=i)
                            else :
                                rfe_clf = clf.best_estimator_
                            
                            rfecv = RFECV(rfe_clf,
                                          scoring = 'roc_auc',
                                          cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                                     test_size=grid_search_test_size,
                                                                     random_state=i),
                                           step = rfe_step,
                                           min_features_to_select = min_rfe)
                                          
                            X_fs = rfecv.fit_transform(X_fs, y_train)
                            X_val_fs = rfecv.transform(X_val_fs)
                            selected_features  = selected_features[rfecv.support_]
                            
                            if plot_rfe :
                                plt.figure()
                                plt.xlabel('Number of features selected')
                                plt.ylabel('Cross validation roc auc score')
                                plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
                                plt.savefig(os.path.join(outdir,'rfe_plot_{}.png'.format(grid_df_row_count)), dpi=600)
                            
                            # redo grid search inner test/val split
                            clf = GridSearchCV(estimator=model, 
                                               param_grid=models_grid[model],
                                               scoring = 'roc_auc',
                                               cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                                         test_size=grid_search_test_size,
                                                                         random_state=i),
                                               n_jobs=n_jobs)
                            
                            clf.fit(X_fs, y_train)
                           
                            
                            y_val_predict_p = clf.best_estimator_.predict_proba(X_val_fs)
                            if y_val_predict_p.shape[1] == 2 :
                                clf_auc = roc_auc_score(y_val, y_val_predict_p[:,1])
                            else :
                                clf_auc = roc_auc_score(y_val, y_val_predict_p)

                            y_train_predict_p = clf.best_estimator_.predict_proba(X_fs)                      
                            if y_train_predict_p.shape[1] == 2 :
                                clf_auc_train = roc_auc_score(y_train, y_train_predict_p[:,1])
                            else :
                                clf_auc_train = roc_auc_score(y_train, y_train_predict_p)

                            y_val_predict = clf.best_estimator_.predict(X_val_fs)
                            clf_acc = accuracy_score(y_val, y_val_predict)
                            clf_bal_acc = balanced_accuracy_score(y_val, y_val_predict)
                            clf_pre = precision_score(y_val, y_val_predict)
                            clf_rec = recall_score(y_val, y_val_predict)
                            
                            # writing pipeline results with rfecv in grid_df                            
                            grid_df.loc[grid_df_row_count, 'hyperparameters with rfecv'] = \
                                str(clf.best_params_)
                            
                            grid_df.loc[grid_df_row_count, 'AUC_train with rfecv'] = round(clf_auc_train, 3)
                            grid_df.loc[grid_df_row_count, 'AUC with rfecv'] = round(clf_auc, 3)
                            grid_df.loc[grid_df_row_count, 'Accuracy with rfecv'] = round(clf_acc, 3)
                            grid_df.loc[grid_df_row_count, 'Balanced_Accuracy with rfecv'] = round(clf_bal_acc, 3)
                            grid_df.loc[grid_df_row_count, 'Precision with rfecv'] = round(clf_pre, 3)
                            grid_df.loc[grid_df_row_count, 'Recall with rfecv'] = round(clf_rec, 3)
                                    
                        else :
                            pass
                        
                        grid_df.loc[grid_df_row_count, 'num_seed'] = int(i)
                        grid_df.loc[grid_df_row_count, 'features'] = str(selected_features)

                        
                        # print to know when a pipeline has been evaluated 
                        print('model', grid_df_row_count+1, '/', len(grid_rows), 'done')
                        grid_df_row_count += 1
    
        # exports of results 
        if outdir != None :
            now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            outpath = os.path.join(outdir,'search_classifiers_{}.xlsx'.format(now))
            grid_df.to_excel(outpath)  
        
        self.results_ = grid_df
        
        print('search_classifiers is finished')
        
        if outdir != None :
            print('results have been written in ',
                  outpath)
        
        print('results are accessible through the results_ attribute')
            
        
        return None



    def search_DNNmodel(self, 
                      outdir,
                      random_iter = range(5),
                      test_size = 0.25,
                      rfe_search_splits = 5,
                      rfe_search_test_size = 0.25,
                      rfe = True,
                      rfe_step = 1,
                      min_rfe = 1,
                      rfe_model = None,
                      rfe_params = None,
                      plot_rfe = False,
                      kernel_initializer = ['he_normal'],
                      activation = ['relu'],
                      dropout_rate = [None],
                      n_output = [1],
                      activation_output = ['sigmoid'],
                      loss = ['binary_crossentropy'],
                      optimizer = ['nadam'],
                      lr = [0.001],
                      metrics = [keras_AUC()],
                      layers = [[32,32]],
                      weight_l2 = [1],
                      batch_norm = [None],
                      epochs = 200,
                      batch_size = 32,
                      class_weight = None,
                      batch_generator = None,
                      callbacks = 'default',
                      monitor = 'val_loss',
                      patience = 10,
                      reduce_lr = 5,
                      min_delta = 0.1,
                      mode = 'min',
                      preprocess_pipelines = {'passthrough':'passthrough'}, 
                      feature_selection_pipelines = {'no_feature_selection':'passthrough'}
                      ) :
        
        """
        Evaluating DNN classifier pipelines through nested train/val split
        """
        
        # creating a df to store each pipeline results
        grid_cols = ['preprocessing', 
                 'feature_selection', 
                 'model', 
                 'hyperparameters',
                 'AUC_train',
                 'Loss_train',
                 'AUC_in_val',
                 'Loss_val',
                 'AUC_out_val',
                 'Loss_out',
                 'RFE',
                 'num_seed',
                 'features']
        
        len_params = len(kernel_initializer)*len(activation)*len(dropout_rate)*len(n_output)*len(activation_output)*len(loss)*len(optimizer)* \
                    len(lr)*len(layers)*len(weight_l2)*len(batch_norm)
        
        grid_rows = range(len(preprocess_pipelines)*len(feature_selection_pipelines)*len_params*len(random_iter))
        
        grid_df =  pd.DataFrame(columns=grid_cols, index=grid_rows)
        
                
        # setting a count for storing info in grid_df
        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        grid_df_row_count = 0
        model_number = 1
        
        # starting the first loop over the random_iter splits/random_state
        for i in random_iter :
            
            # split between train and validation data
            split = StratifiedShuffleSplit(n_splits=1, 
                                           test_size=test_size, 
                                           random_state=i)
            
            for train_index, val_index in split.split(self.train_x, self.train_y):
                X_train = self.train_x.loc[train_index]
                X_val = self.train_x.loc[val_index]
                y_train = self.train_y.loc[train_index]
                y_val = self.train_y.loc[val_index]
            
            X_train.reset_index(drop=True, inplace=True)
            X_val.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_val.reset_index(drop=True, inplace=True)

                        
            # Looping over each preprocessing pipeline submitted 
            for preprocess_pipeline in preprocess_pipelines.values() :
        
                selected_features = self.features.copy()

                if preprocess_pipeline == 'passthrough' :
                    X_p = X_train.values
                    X_val_p = X_val.values
                else :
                    X_p = preprocess_pipeline.fit_transform(X_train)
                    X_val_p = preprocess_pipeline.transform(X_val)    
                    
                    try :
                        preprocess_pipeline.named_transformers_.num_pipe['polynomial']
                    except KeyError :
                        pass
                    else :
                        selected_features = preprocess_pipeline.named_transformers_.num_pipe['polynomial'].get_feature_names(selected_features)
                        selected_features = pd.Series(selected_features)
                
                # Looping over each feature selection(s) :
                for feature_selection in feature_selection_pipelines.values() :
            
                    if feature_selection == 'passthrough' :
                        X_fs = X_p
                        X_val_fs = X_val_p
                        
                    else : 
                        X_fs = feature_selection.fit_transform(X_p, y_train)
                        X_val_fs = feature_selection.transform(X_val_p)
                        selected_features  = selected_features[feature_selection.support_]
                        
                
                    if rfe :
                        
                        if rfe_model == 'XGBoost' :
                            rfe_clf = XGBClassifier(n_jobs=-2, random_state=i).set_params(**rfe_params)
                            rfecv = RFECV(rfe_clf,
                                          scoring = 'roc_auc',
                                          cv=StratifiedShuffleSplit(n_splits=rfe_search_splits, 
                                                                     test_size=rfe_search_test_size,
                                                                     random_state=i),
                                           step = rfe_step,
                                           min_features_to_select = min_rfe)
                                          
                            X_fs = rfecv.fit_transform(X_fs, y_train)
                            X_val_fs = rfecv.transform(X_val_fs)
                            selected_features  = selected_features[rfecv.support_]

                            if plot_rfe :
                                plt.figure()
                                plt.xlabel('Number of features selected')
                                plt.ylabel('Cross validation roc auc score')
                                plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
                                plt.savefig(os.path.join(outdir,'rfe_plot_{}.png'.format(grid_df_row_count)), dpi=600)


                        elif rfe_model == 'DNN' :
                            pass
                        else :
                            raise ValueError 
                            print('only "XGBoost" or "DNN" are valid models for RFE at the moment')
                        
                               
                    # nested val set for DNN training
                    split = StratifiedShuffleSplit(n_splits=1, 
                                                   test_size=test_size, 
                                                   random_state=i)
                    
                    for train_index, val_index in split.split(X_fs, y_train):
                        X_nes_train = X_fs[train_index]
                        X_nes_val = X_fs[val_index]
                        y_nes_train = y_train[train_index]
                        y_nes_val = y_train[val_index]
                    
                    if class_weight :
                        class_weight = dict()
                        for num_class in range(len(pd.value_counts(y_nes_train))) :
                            class_weight[int(num_class)] = (1 / (np.unique(y_nes_train, return_counts=True)[1][num_class]))*(len(y_nes_train))/2.0
                    else :
                        class_weight = None
        
                    # Looping over each model hyperparameters
                    for kernel_init in kernel_initializer :
                        for activate in activation :
                            for dropout in dropout_rate :
                                for output in n_output :
                                    for activate_output in activation_output :
                                        for los in loss :
                                            for optim in optimizer :
                                                for lr_ in lr :
                                                    for layer in layers :
                                                        for l2weight in weight_l2 :
                                                            for batch_n in batch_norm :
                                                                
                                                                K.clear_session()
                                                                                                                                                                                     
                                                                if ((rfe_model == 'DNN') & (rfe == True)) :
                                                                    
                                                                    base_mask = {feat:True for feat in selected_features}
                                                                    
                                                                    rfe_dict_auc = dict()
                                                                    rfe_dict_auc_out = dict()
                                                                    rfe_dict_auc_train = dict()
                                                                    rfe_dict_loss = dict()
                                                                    rfe_dict_loss_out = dict()
                                                                    rfe_dict_loss_train = dict()
                                                                                    
                                                                    for count in range(X_nes_train.shape[1], min_rfe, -1) :
                                                                                                                                                
                                                                        if callbacks == 'default' :
                                                                            callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                                                                         ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                                                                        
                                                                        elif callbacks == 'full' :
                                                                            
                                                                            savedir = os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(grid_df_row_count)), 'model.h5')
                                                                            logdir = os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(grid_df_row_count)))
                                                                
                                                                            callbacks = [ModelCheckpoint(filepath=savedir, monitor=monitor, save_best_only=True, mode=mode),
                                                                                         TensorBoard(log_dir=logdir),
                                                                                         EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                                                                         ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                                                                
                                                                        else :
                                                                            callbacks = callbacks

                                                                        X_tmp = X_nes_train[:,list(base_mask.values())]
                                                                        X_val_tmp = X_nes_val[:,list(base_mask.values())]
                                                                        X_outer_val = X_val_fs[:,list(base_mask.values())]
                                                                        
                                                                        clf = make_DNNmodel(input_dim = X_tmp.shape[1],
                                                                               kernel_initializer = kernel_init,
                                                                               kernel_seed = i,
                                                                               activation = activate,
                                                                               dropout_rate = dropout,
                                                                               n_output = output,
                                                                               activation_output = activate_output,
                                                                               loss = los,
                                                                               optimizer = optim,
                                                                               lr = lr_,
                                                                               metrics = [keras_AUC()],
                                                                               layers = layer,
                                                                               weight_l2 = l2weight,
                                                                               batch_norm = batch_n) 
                                                            
                                                                        if batch_generator == None :
                                                                            clf.fit(X_tmp, y_nes_train, 
                                                                                    validation_data=(X_val_tmp, y_nes_val), 
                                                                                    epochs = epochs, 
                                                                                    callbacks=callbacks, 
                                                                                    class_weight=class_weight,
                                                                                    batch_size=batch_size
                                                                                    )
# =============================================================================
#                                                                         else :
#                                                                             training_generator = BalancedBatchGenerator(X_tmp,
#                                                                                                                         y_nes_train,
#                                                                                                                         batch_size=batch_size,
#                                                                                                                         random_state=i)
#                                                                             clf.fit(training_generator, 
#                                                                                     validation_data=(X_val_tmp, y_nes_val), 
#                                                                                     epochs = epochs, 
#                                                                                     callbacks=callbacks, 
#                                                                                     class_weight=class_weight
#                                                                                     )
#                                                                         
# =============================================================================
                                                                        # storing model's auc and loss for future selection of the best subset of features
                                                                        y_val_DNN = clf.predict(X_val_tmp)
                                                                        DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_val, y_val_DNN)
                                                                        DNN_auc = auc(DNN_fpr, DNN_tpr)
                                                                        rfe_dict_auc[str(count)] = round(DNN_auc, 4)

                                                                        y_val_out_DNN = clf.predict(X_outer_val)
                                                                        DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_val, y_val_out_DNN)
                                                                        DNN_auc_out = auc(DNN_fpr, DNN_tpr)
                                                                        rfe_dict_auc_out[str(count)] = round(DNN_auc_out, 4)

                                                                        y_val_train_DNN = clf.predict(X_tmp)
                                                                        DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_train, y_val_train_DNN)
                                                                        DNN_auc_train = auc(DNN_fpr, DNN_tpr)
                                                                        rfe_dict_auc_train[str(count)] = round(DNN_auc_train, 4)

                                                                        loss_val = clf.evaluate(X_val_tmp, y_nes_val)[0]
                                                                        rfe_dict_loss[str(count)] = round(loss_val ,4)

                                                                        loss_out = clf.evaluate(X_outer_val, y_val)[0]
                                                                        rfe_dict_loss_out[str(count)] = round(loss_out,4)

                                                                        loss_train = clf.evaluate(X_tmp, y_nes_train)[0]
                                                                        rfe_dict_loss_train[str(count)] = round(loss_train,4)

                                                                        # computing feature importances and removing features
                                                                        def score_importance(X_score, y_score):
                                                    
                                                                            if isinstance(y_score, pd.Series) :
                                                                                y_score = y_score.values
                                                                            if isinstance(X_score, pd.DataFrame) :
                                                                                X_score = X_score.values
                                                                        
                                                                            y_score_ = clf.predict(X_score)
                                                                            # we take the mean loss
                                                                            # and negate it so the highest loss = the lowest score
                                                                            return -K.eval(K.mean(binary_crossentropy(tf.convert_to_tensor(y_score.reshape(-1,1), np.float32), tf.convert_to_tensor(y_score_[:,:], np.float32))))
                                                    
                                                                        base_score, score_decreases = get_score_importances(score_importance, X_val_tmp, y_nes_val, n_iter=20, random_state = i)
                                                                        feature_importances = list(np.mean(score_decreases, axis=0))
                                                                        feat_to_remove = feature_importances.index(min(feature_importances))
                                                                        key_to_remove = list({i:j for i, j in base_mask.items() if j == True}.keys())[feat_to_remove]
                                                                        base_mask[key_to_remove] = False
                                                                        
                                                                        K.clear_session()
                                                                        print(abs(count-X_nes_train.shape[1]-1), '/', ((X_nes_train.shape[1])-min_rfe), 'rfe done')

                                                                        # dnn params
                                                                        DNN_params = {
                                                                                'kernel_initializer' : kernel_init,
                                                                                'activation' : activate,
                                                                                'dropout_rate' : dropout,
                                                                                'n_output' : output,
                                                                                'activation_output' : activate_output,
                                                                                'loss' : los,
                                                                                'optimizer' : optim,
                                                                                'learning_rate' : lr_,
                                                                                'layers' : layer,
                                                                                'weight_l2' : l2weight,
                                                                                'batch_norm' : batch_n,
                                                                                'epochs' : epochs,
                                                                                'batch_size' : batch_size,
                                                                                'class_weight' : class_weight,
                                                                                'batch_generator' : batch_generator,
                                                                                'callbacks':str(callbacks)
                                                                                }
                                                
                                                                        # writing pipeline results in grid_df
                                                                        grid_df.loc[grid_df_row_count, 'preprocessing'] = \
                                                                            list(preprocess_pipelines.keys())[list(preprocess_pipelines.values()).index(preprocess_pipeline)]
                                                                       
                                                                        grid_df.loc[grid_df_row_count, 'feature_selection'] = \
                                                                            list(feature_selection_pipelines.keys())[list(feature_selection_pipelines.values()).index(feature_selection)]
                                                                        
                                                                        grid_df.loc[grid_df_row_count, 'model'] = \
                                                                            str('DNN')
                                                                        
                                                                        grid_df.loc[grid_df_row_count, 'hyperparameters'] = \
                                                                            str(DNN_params)
                                                                        
                                                                        grid_df.loc[grid_df_row_count, 'AUC_train'] = round(DNN_auc_train, 4)

                                                                        grid_df.loc[grid_df_row_count, 'Loss_train'] = round(loss_train, 4)

                                                                        grid_df.loc[grid_df_row_count, 'AUC_in_val'] = round(DNN_auc, 4)

                                                                        grid_df.loc[grid_df_row_count, 'Loss_val'] = round(loss_val, 4)
    
                                                                        grid_df.loc[grid_df_row_count, 'AUC_out_val'] = round(DNN_auc_out, 4)
                                                
                                                                        grid_df.loc[grid_df_row_count, 'Loss_out'] = round(loss_out, 4)

                                                                        grid_df.loc[grid_df_row_count, 'RFE'] = str(rfe_model)
                                                                        
                                                                        grid_df.loc[grid_df_row_count, 'num_seed'] = int(i)

                                                                        grid_df.loc[grid_df_row_count, 'rfe_number'] = len([key for key, value in base_mask.items() if value ==True])
                                                                                    
                                                                        grid_df.loc[grid_df_row_count, 'features'] = str([key for key, value in base_mask.items() if value ==True])

                                                                        grid_df.loc[grid_df_row_count, 'feature_mask'] = str([value for key, value in base_mask.items()])

                                                                        # print to know when a rfe has been evaluated 
                                                                        print('model', model_number, '/', len(grid_rows), 'done')
                                                                        grid_df_row_count += 1
   
                                                                    
                                                                    # plotting rfe if true
                                                                    if plot_rfe :
                                                                        plt.figure()
                                                                        plt.subplot(211)
                                                                        plt.ylabel('roc auc score')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_auc), -1), list(rfe_dict_auc.values()), 'b--', label='validation set')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_auc_out), -1), list(rfe_dict_auc_out.values()), 'b:', label='outer validation set')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_auc_train), -1), list(rfe_dict_auc_train.values()), 'b-', label='training set')
                                                                        plt.legend(loc="lower right")
                                                                        plt.subplot(212)
                                                                        plt.xlabel('Number of features selected')
                                                                        plt.ylabel('loss')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_loss), -1), list(rfe_dict_loss.values()), 'y--', label='validation set')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_loss_out), -1), list(rfe_dict_loss_out.values()), 'y:', label='outer validation set')
                                                                        plt.plot(range(X_nes_train.shape[1], X_nes_train.shape[1]-len(rfe_dict_loss_train), -1), list(rfe_dict_loss_train.values()), 'y-', label='training set')
                                                                        plt.legend(loc="lower right")
                                                                        plt.savefig(os.path.join(outdir,'rfe_plot_DNNtrained_{}.png'.format(i)), dpi=600)
                                                    

                                                                # or training a model without RFE
                                                                else :
                                                                    
                                                                    if callbacks == 'default' :
                                                                        callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                                                                     ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                                                                    
                                                                    elif callbacks == 'full' :
                                                                        
                                                                        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                                                                        savedir = os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(grid_df_row_count)), 'model.h5')
                                                                        logdir = os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(grid_df_row_count)))
                                                            
                                                                        callbacks = [ModelCheckpoint(filepath=savedir, monitor=monitor, save_best_only=True, mode=mode),
                                                                                     TensorBoard(log_dir=logdir),
                                                                                     EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                                                                     ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                                                            
                                                                    else :
                                                                        callbacks = callbacks

                                                                    clf = make_DNNmodel(input_dim = X_nes_train.shape[1],
                                                                           kernel_initializer = kernel_init,
                                                                           kernel_seed = i,
                                                                           activation = activate,
                                                                           dropout_rate = dropout,
                                                                           n_output = output,
                                                                           activation_output = activate_output,
                                                                           loss = los,
                                                                           optimizer = optim,
                                                                           lr = lr_,
                                                                           metrics = [keras_AUC()],
                                                                           layers = layer,
                                                                           weight_l2 = l2weight,
                                                                           batch_norm = batch_n) 

                                                                    if batch_generator == None :
                                                                        clf.fit(X_nes_train, y_nes_train, 
                                                                                validation_data=(X_nes_val, y_nes_val), 
                                                                                epochs = epochs, 
                                                                                callbacks=callbacks, 
                                                                                class_weight=class_weight,
                                                                                batch_size=batch_size)
# =============================================================================
#                                                                     else :
#                                                                         training_generator = BalancedBatchGenerator(X_nes_train,
#                                                                                                                     y_nes_train,
#                                                                                                                     batch_size=batch_size,
#                                                                                                                     random_state=i)
#                                                                         clf.fit(training_generator, 
#                                                                                 validation_data=(X_nes_val, y_nes_val), 
#                                                                                 epochs = epochs, 
#                                                                                 callbacks=callbacks, 
#                                                                                 class_weight=class_weight
#                                                                                 )
# 
# =============================================================================
                                                                        
                                                                    # computing auc on holdout val data
                                                                    y_val_DNN = clf.predict(X_nes_val)
                                                                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_val, y_val_DNN)
                                                                    DNN_auc = auc(DNN_fpr, DNN_tpr)

                                                                    y_val_DNN_out = clf.predict(X_val_fs)
                                                                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_val, y_val_DNN_out)
                                                                    DNN_auc_out = auc(DNN_fpr, DNN_tpr)

                                                                    y_val_train_DNN = clf.predict(X_nes_train)
                                                                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_train, y_val_train_DNN)
                                                                    DNN_auc_train = auc(DNN_fpr, DNN_tpr)

                                                                    loss_val = clf.evaluate(X_nes_val, y_nes_val)[0]

                                                                    loss_out = clf.evaluate(X_val_fs, y_val)[0]

                                                                    loss_train = clf.evaluate(X_nes_train, y_nes_train)[0]
                                                                    
                                                                    K.clear_session()

                                                                    # dnn params
                                                                    DNN_params = {
                                                                            'kernel_initializer' : kernel_init,
                                                                            'activation' : activate,
                                                                            'dropout_rate' : dropout,
                                                                            'n_output' : output,
                                                                            'activation_output' : activate_output,
                                                                            'loss' : los,
                                                                            'optimizer' : optim,
                                                                            'learning_rate' : lr_,
                                                                            'layers' : layer,
                                                                            'weight_l2' : l2weight,
                                                                            'batch_norm' : batch_n,
                                                                            'epochs' : epochs,
                                                                            'batch_size' : batch_size,
                                                                            'class_weight' : class_weight,
                                                                            'batch_generator' : batch_generator,
                                                                            'callbacks':str(callbacks)
                                                                            }
                                            
                                                                    # writing pipeline results in grid_df
                                                                    grid_df.loc[grid_df_row_count, 'preprocessing'] = \
                                                                        list(preprocess_pipelines.keys())[list(preprocess_pipelines.values()).index(preprocess_pipeline)]
                                                                   
                                                                    grid_df.loc[grid_df_row_count, 'feature_selection'] = \
                                                                        list(feature_selection_pipelines.keys())[list(feature_selection_pipelines.values()).index(feature_selection)]
                                                                    
                                                                    grid_df.loc[grid_df_row_count, 'model'] = \
                                                                        str('DNN')
                                                                    
                                                                    grid_df.loc[grid_df_row_count, 'hyperparameters'] = \
                                                                        str(DNN_params)
                                                                    
                                                                    grid_df.loc[grid_df_row_count, 'AUC_train'] = round(DNN_auc_train, 4)

                                                                    grid_df.loc[grid_df_row_count, 'Loss_train'] = round(loss_train, 4)

                                                                    grid_df.loc[grid_df_row_count, 'AUC_in_val'] = round(DNN_auc, 4)

                                                                    grid_df.loc[grid_df_row_count, 'Loss_val'] = round(loss_val, 4)

                                                                    grid_df.loc[grid_df_row_count, 'AUC_out_val'] = round(DNN_auc_out, 4)
                                            
                                                                    grid_df.loc[grid_df_row_count, 'Loss_out'] = round(loss_out, 4)
                                            
                                                                    grid_df.loc[grid_df_row_count, 'RFE'] = str(rfe_model)
                                                                    
                                                                    grid_df.loc[grid_df_row_count, 'num_seed'] = int(i)

                                                                    grid_df.loc[grid_df_row_count, 'num_features'] = len(list(selected_features))

                                                                    grid_df.loc[grid_df_row_count, 'features'] = str(list(selected_features))
                                                                        
                                                                    
                                                                # print to know when a pipeline has been evaluated 
                                                                print('model', model_number, '/', len(grid_rows), 'done')
                                                                grid_df_row_count += 1
                                                                model_number += 1
                                                                    
        # formatting the results (calculating mean_AUC over each iteration)
        if ((rfe_model == 'DNN') & (rfe == True)) :
            
            tmp = np.unique(grid_df['rfe_number'])
    
            for i in range(len(tmp)) :
        
                mean_auc = round(grid_df.loc[grid_df[grid_df['rfe_number'] == tmp[i]].index, 'AUC_out_val'].mean(), 3)
    
                grid_df.loc[grid_df[grid_df['rfe_number'] == tmp[i]].index, 'mean_AUC_val'] = mean_auc
        
        else :
            
            tmp = np.unique(grid_df['hyperparameters'])
    
            for i in range(len(tmp)) :
        
                mean_auc = round(grid_df.loc[grid_df[grid_df['hyperparameters'] == tmp[i]].index, 'AUC_out_val'].mean(), 3)
    
                grid_df.loc[grid_df[grid_df['hyperparameters'] == tmp[i]].index, 'mean_AUC_val'] = mean_auc
        
        # exports of results 
        if outdir != None :
            now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            outpath = os.path.join(outdir, 'search_DNNclassifiers_{}.xlsx'.format(now))
            grid_df.to_excel(outpath)  
            print('results have been written in ',
                  outpath)
        
        self.results_ = grid_df
        
        print('search_DNNclassifiers is finished')
                
        print('results are accessible through the results_ attribute')
            
        
        return None




    def search_additive_DNNmodel(self, 
                      outdir,
                      bootstrap = range(10),
                      test_size = 0.25,
                      max_patience = 10,
                      select_criterion = 'out_val_auc_median',
                      plot = False,
                      kernel_initializer = 'he_normal',
                      activation = 'relu',
                      dropout_rate = None,
                      n_output = 1,
                      activation_output = 'sigmoid',
                      loss = 'binary_crossentropy',
                      optimizer = 'nadam',
                      lr = 0.001,
                      metrics = [keras_AUC()],
                      layers = [100, 50],
                      weight_l2 = None,
                      batch_norm = None,
                      epochs = 200,
                      batch_size = 32,
                      class_weight = None,
                      batch_generator = None,
                      callbacks = 'default',
                      monitor = 'val_loss',
                      patience = 10,
                      reduce_lr = 5,
                      min_delta = 0.1,
                      mode = 'min',
                      preprocessing = 'passthrough', 
                      feature_selection = 'passthrough'
                      ) :
        
        """
        Evaluating a DNN classifier pipeline with feature being added one by one
        Evaluation of the best subset of features is done through multiple bootstrapped and 
        a dual, inner and outer, validation set
        """
        
        # creating a df to store each pipeline results
        grid_cols = ['preprocessing', 
                 'feature_selection', 
                 'model', 
                 'hyperparameters',
                 'AUC_median_train',
                 'AUC_std_train',
                 'loss_median_train',
                 'loss_std_train',
                 'AUC_median_in_val',
                 'AUC_std_in_val',
                 'loss_median_in_val',
                 'loss_std_in_val',
                 'AUC_median_out_val',
                 'AUC_std_out_val',
                 'loss_median_out_val',
                 'loss_std_out_val',
                 'num_bootstrap_seed',
                 'num_features',
                 'features',
                 'feature_mask']
        
        len_df = range(max_patience)        
        grid_df =  pd.DataFrame(columns=grid_cols, index=len_df)
                        
        # setting a count for storing info in grid_df after each feature addition
        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')

        # applying feature selection if stated :    
        if feature_selection == 'passthrough' :
            pass
            
        else : 
            self.train_x = feature_selection.fit_transform(self.train_x, self.train_y)
            self.features  = self.features[feature_selection.support_]
            self.train_x = pd.DataFrame(data=self.train_x, columns = self.features)

        # loop while we have not met the criteria for number of features added:            
        added_feat = 0
        base_mask = {feat:False for feat in self.features}
        
        while added_feat < max_patience :
            
            # take a feature to add and bootstrap different models with it:
            features_medians_cols = ['train_auc_median',
                                     'train_auc_std',
                                     'in_val_auc_median', 
                                     'in_val_auc_std',
                                     'out_val_auc_median',
                                     'out_val_auc_std',
                                     'train_loss_median',
                                     'train_loss_std',
                                     'in_val_loss_median',
                                     'in_val_loss_std',
                                     'out_val_loss_median',
                                     'out_val_loss_std']
            features_medians = pd.DataFrame(index = self.features, columns = features_medians_cols, dtype='float64')
            
            feat_num = 0
            len_feat = len([key for key, value in base_mask.items() if value == False])
            seeds = [seed+added_feat*30 for seed in bootstrap]
            
            for feat in [key for key, value in base_mask.items() if value == False] :
            
                # setting a test mask for features to try :
                test_mask = base_mask.copy()
                test_mask[feat] = True
                
                X_train_tmp = self.train_x.loc[:,list(test_mask.values())]
                                
                # lists to store DNN results on bootstrapped data :
                train_aucs = list()   
                train_losses = list()
                in_val_aucs = list()
                in_val_losses = list()
                out_val_aucs = list()
                out_val_losses = list()
                
                # Doing the different bootstrap
                bootstrap_num = 1
                for seed in seeds :
                    # split between train and validation data
                    split = StratifiedShuffleSplit(n_splits=1, 
                                                   test_size=test_size, 
                                                   random_state=seed)
                    
                    for train_index, val_index in split.split(X_train_tmp, self.train_y):
                        X_train = X_train_tmp.loc[train_index]
                        X_val = X_train_tmp.loc[val_index]
                        y_train = self.train_y.loc[train_index]
                        y_val = self.train_y.loc[val_index]
                    
                    X_train.reset_index(drop=True, inplace=True)
                    X_val.reset_index(drop=True, inplace=True)
                    y_train.reset_index(drop=True, inplace=True)
                    y_val.reset_index(drop=True, inplace=True)
        
                    # Looping over each preprocessing pipeline submitted         
                    selected_features = self.features.copy()
            
                    if preprocessing == 'passthrough' :
                        X_p = X_train.values
                        X_val_p = X_val.values
                    else :
                        X_p = preprocessing.fit_transform(X_train)
                        X_val_p = preprocessing.transform(X_val)    
                       
                        try :
                            preprocessing.named_transformers_.num_pipe['polynomial']
                        except KeyError :
                            pass
                        else :
                            selected_features = preprocessing.named_transformers_.num_pipe['polynomial'].get_feature_names(selected_features)
                            selected_features = pd.Series(selected_features)
        
                                
                    # resplit to obtain an inner and a outer val
                    split = StratifiedShuffleSplit(n_splits=1, 
                                                   test_size=test_size, 
                                                   random_state=seed)
                    
                    for train_index, val_index in split.split(X_p, y_train):
                        X_nes_train = X_p[train_index]
                        X_nes_val = X_p[val_index]
                        y_nes_train = y_train[train_index]
                        y_nes_val = y_train[val_index]
                    
                        
                
                    # Training the DNN
                    K.clear_session()
                                                                                                                                         
                    if class_weight :
                        class_weight = dict()
                        for num_class in range(len(pd.value_counts(y_nes_train))) :
                            class_weight[int(num_class)] = (1 / (np.unique(y_nes_train, return_counts=True)[1][num_class]))*(len(y_nes_train))/2.0
                    else :
                        class_weight = None
                        
                        base_mask = {feat:True for feat in selected_features}
                                                    
                            
                    if callbacks == 'default' :
                        callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                     ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                            
                    elif callbacks == 'full' :
                        
                        savedir = os.path.join(outdir, 'run-{}'.format(now), 'add_{}'.format(str(added_feat)), 'model.h5')
                        logdir = os.path.join(outdir, 'run-{}'.format(now), 'add_{}'.format(str(added_feat)))
            
                        callbacks = [ModelCheckpoint(filepath=savedir, monitor=monitor, save_best_only=True, mode=mode),
                                     TensorBoard(log_dir=logdir),
                                     EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                     ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
                    
                    else :
                        callbacks = callbacks
        
                    clf = make_DNNmodel(input_dim = X_nes_train.shape[1],
                           kernel_initializer = kernel_initializer,
                           kernel_seed = seed,
                           activation = activation,
                           dropout_rate = dropout_rate,
                           n_output = n_output,
                           activation_output = activation_output,
                           loss = loss,
                           optimizer = optimizer,
                           lr = lr,
                           metrics = [keras_AUC()],
                           layers = layers,
                           weight_l2 = weight_l2,
                           batch_norm = batch_norm) 
        
                    if batch_generator == None :
                        clf.fit(X_nes_train, y_nes_train, 
                                validation_data=(X_nes_val, y_nes_val), 
                                epochs = epochs, 
                                callbacks=callbacks, 
                                class_weight=class_weight,
                                batch_size=batch_size
                                )
# =============================================================================
#                     else :
#                         training_generator = BalancedBatchGenerator(X_nes_train,
#                                                                     y_nes_train,
#                                                                     batch_size=batch_size,
#                                                                     random_state=seed)
#                         clf.fit(training_generator, 
#                                 validation_data=(X_nes_val, y_nes_val), 
#                                 epochs = epochs, 
#                                 callbacks=callbacks, 
#                                 class_weight=class_weight
#                                 )
#                             
# =============================================================================
                    # storing model's auc and loss for future selection of the best subset of features
                    y_val_DNN = clf.predict(X_nes_val)
                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_val, y_val_DNN)
                    DNN_auc = auc(DNN_fpr, DNN_tpr)
                    in_val_aucs.append(round(DNN_auc, 4))
    
                    y_val_out_DNN = clf.predict(X_val_p)
                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_val, y_val_out_DNN)
                    DNN_auc_out = auc(DNN_fpr, DNN_tpr)
                    out_val_aucs.append(round(DNN_auc_out, 4))
    
                    y_val_train_DNN = clf.predict(X_nes_train)
                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_nes_train, y_val_train_DNN)
                    DNN_auc_train = auc(DNN_fpr, DNN_tpr)
                    train_aucs.append(round(DNN_auc_train, 4))
    
                    loss_val = clf.evaluate(X_nes_val, y_nes_val)[0]
                    in_val_losses.append(round(loss_val ,4))
    
                    loss_out = clf.evaluate(X_val_p, y_val)[0]
                    out_val_losses.append(round(loss_out ,4))
    
                    loss_train = clf.evaluate(X_nes_train, y_nes_train)[0]
                    train_losses.append(round(loss_train ,4))
                                  
                    K.clear_session()
                        
                    print('{} feature tested out of {}.'.format(feat_num, len_feat))
                    print('{} bootstrap tested out of {}.'.format(bootstrap_num, len(bootstrap)))
                    print('{} feature added out of {}.'.format(added_feat, max_patience))
                    
                    bootstrap_num += 1
                    
                # Computing and storing median results for one feature
                features_medians.loc[feat, 'train_auc_median'] = np.median(train_aucs)    
                features_medians.loc[feat, 'train_auc_std'] = np.std(train_aucs)    
                features_medians.loc[feat, 'train_loss_median'] = np.median(train_losses)    
                features_medians.loc[feat, 'train_loss_std'] = np.std(train_losses)    
                features_medians.loc[feat, 'in_val_auc_median'] = np.median(in_val_aucs)    
                features_medians.loc[feat, 'in_val_auc_std'] = np.std(in_val_aucs)    
                features_medians.loc[feat, 'in_val_loss_median'] = np.median(in_val_losses)    
                features_medians.loc[feat, 'in_val_loss_std'] = np.std(in_val_losses)    
                features_medians.loc[feat, 'out_val_auc_median'] = np.median(out_val_aucs)    
                features_medians.loc[feat, 'out_val_auc_std'] = np.std(out_val_aucs)    
                features_medians.loc[feat, 'out_val_loss_median'] = np.median(out_val_losses)    
                features_medians.loc[feat, 'out_val_loss_std'] = np.std(out_val_losses)    

                feat_num += 1
                                
                
            # Finding the best feature to add, rinse and repeat by updating base mask and added_feat
            best_feat = features_medians[select_criterion].idxmax()
            base_mask[best_feat] = True
                         
            # adding info in grid_df
            # dnn params
            DNN_params = {
                    'kernel_initializer' : kernel_initializer,
                    'activation' : activation,
                    'dropout_rate' : dropout_rate,
                    'n_output' : n_output,
                    'activation_output' : activation_output,
                    'loss' : loss,
                    'optimizer' : optimizer,
                    'lr' : lr,
                    'layers' : layers,
                    'weight_l2' : weight_l2,
                    'batch_norm' : batch_norm,
                    'epochs' : epochs,
                    'batch_size' : batch_size,
                    'class_weight' : class_weight,
                    'batch_generator' : batch_generator,
                    'callbacks':str(callbacks)
                    }

            # writing pipeline results in grid_df
            grid_df.loc[added_feat, 'preprocessing'] = \
                str(preprocessing)
           
            grid_df.loc[added_feat, 'feature_selection'] = \
                str(feature_selection)
            
            grid_df.loc[added_feat, 'model'] = \
                str('DNN')
            
            grid_df.loc[added_feat, 'hyperparameters'] = \
                str(DNN_params)
            
            grid_df.loc[added_feat, 'AUC_median_train'] = features_medians.loc[best_feat, 'train_auc_median']

            grid_df.loc[added_feat, 'AUC_std_train'] = features_medians.loc[best_feat, 'train_auc_std']

            grid_df.loc[added_feat, 'loss_median_train'] = features_medians.loc[best_feat, 'train_loss_median']

            grid_df.loc[added_feat, 'loss_std_train'] = features_medians.loc[best_feat, 'train_loss_std']

            grid_df.loc[added_feat, 'AUC_median_in_val'] = features_medians.loc[best_feat, 'in_val_auc_median']

            grid_df.loc[added_feat, 'AUC_std_in_val'] = features_medians.loc[best_feat, 'in_val_auc_std']
                        
            grid_df.loc[added_feat, 'loss_median_in_val'] = features_medians.loc[best_feat, 'in_val_loss_median']

            grid_df.loc[added_feat, 'loss_std_in_val'] = features_medians.loc[best_feat, 'in_val_loss_std']

            grid_df.loc[added_feat, 'AUC_median_out_val'] = features_medians.loc[best_feat, 'out_val_auc_median']

            grid_df.loc[added_feat, 'AUC_std_out_val'] = features_medians.loc[best_feat, 'out_val_auc_std']

            grid_df.loc[added_feat, 'loss_median_out_val'] = features_medians.loc[best_feat, 'out_val_loss_median']
            
            grid_df.loc[added_feat, 'loss_std_out_val'] = features_medians.loc[best_feat, 'out_val_loss_std']

            grid_df.loc[added_feat, 'num_bootstrap_seed'] = len(bootstrap)

            grid_df.loc[added_feat, 'num_features'] = len([key for key, value in base_mask.items() if value ==True])
                        
            grid_df.loc[added_feat, 'features'] = str([key for key, value in base_mask.items() if value ==True])

            grid_df.loc[added_feat, 'feature_mask'] = str([value for key, value in base_mask.items()])
               
            # increasing the loop to add another feature if allowed
            added_feat+=1

            # plotting rfe if true (after each feature added)
            if plot :
                plt.figure()
                plt.subplot(211)
                plt.ylabel('roc auc score')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['AUC_median_in_val']), 'b--', label='validation set')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['AUC_median_out_val']), 'b:', label='outer validation set')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['AUC_median_train']), 'b-', label='training set')
                plt.legend(loc="lower right")
                plt.subplot(212)
                plt.xlabel('Number of features selected')
                plt.ylabel('loss')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['loss_median_in_val']), 'y--', label='validation set')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['loss_median_out_val']), 'y:', label='outer validation set')
                plt.plot([i+1 for i in range(max_patience)], list(grid_df['loss_median_train']), 'y-', label='training set')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(outdir,'additive_plot_DNN_{}.png'.format(now)), dpi=600)
                                                    
                                                    
            # exports of results for each feature
            if outdir != None :
                now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                outpath = os.path.join(outdir, 'search_DNNclassifiers_{}.xlsx'.format(now))
                grid_df.to_excel(outpath)  
                print('results have been written in ',
                      outpath)
        
        self.results_ = grid_df
        
        print('search_DNNclassifiers is finished')
                
        print('results are accessible through the results_ attribute')
            
        
        return None

