# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:30:42 2020

@author: Xavier Dieu

===============================================================================
FUNCTIONS AND CLASSES TO TRAIN MODELS FOR EZML
===============================================================================


"""

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import joblib
import shap
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc, roc_curve
from xgboost import XGBClassifier
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
#from imblearn.keras import BalancedBatchGenerator # deprecated
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC as keras_AUC
from ezml.model_maker import make_DNNmodel


# FUNCTIONS


# CLASSES

class Mask_selector(BaseEstimator, TransformerMixin) :
    """
    custom transformer for DNNrfe to use in train_DNNclassifier
    """
    
    def __init__(self, mask) :
        self.support_ = mask
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        return X[:, self.support_]


class EzPipeline :
    
    """
    Easy Machine Learning pipeline class
    """
    
    
    def __init__(self, train_data, train_target, test_data, test_target) :
        self.train_x = train_data
        self.test_x = test_data
        self.train_y = train_target
        self.test_y = test_target
        self.selected_feats_ = None
        self.validation_data_ = None
        self.val_y = None
    
    
    def train_classifier(self, 
                      model_grid,
                      feature_names,
                      outdir,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      clf_params = None,
                      grid_search = False,
                      grid_search_splits = 5,
                      grid_search_test_size = 0.25,
                      rfe = True,
                      rfe_step = 1,
                      min_rfe = 1,
                      rfe_model = None,
                      rfe_params = None,
                      plot_rfe = False,
                      n_jobs = -2,
                      preprocess = 'passthrough', 
                      feature_selection = 'passthrough'
                      ) :
        
        """
        Training a classifier pipeline 
        """

        # feature_names check
        if isinstance(feature_names, pd.Series) :
            selected_feats = feature_names
        else :
             selected_feats = pd.Series(feature_names)           

                
        # Applying designated preprocessing
        if preprocess == 'passthrough' :
            self.train_x = self.train_x.values         
        else :
            self.train_x = preprocess.fit_transform(self.train_x)

            try :
                preprocess.named_transformers_.num_pipe['polynomial']
            except KeyError :
                pass
            else :
                selected_feats = preprocess.named_transformers_.num_pipe['polynomial'].get_feature_names(selected_feats)
                selected_feats = pd.Series(selected_feats)


        # Under sampling if not none :
        if under_sampling == 'random' :
            self.train_x, self.train_y = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=random_seed).fit_resample(self.train_x, self.train_y) 
        
        if under_sampling == 'ENN' :
            self.train_x, self.train_y = EditedNearestNeighbours().fit_resample(self.train_x, self.train_y) 
                
                
        # Applying feature selection or not    
        if feature_selection == 'passthrough' :
            pass
            
        else : 
            self.train_x = feature_selection.fit_transform(self.train_x, self.train_y)
            selected_feats  = feature_names[feature_selection.support_]
    
                                                
        # fitting model (with or without grid search)
        # first, checking that only one model is submitted
        if len(model_grid) != 1 :
            raise ValueError 
            print('Model_grid argument should only be of length 1 (one model)')
        
        if grid_search :
            clf = GridSearchCV(estimator=list(model_grid.keys())[0], 
                               param_grid=list(model_grid.values())[0],
                               scoring = 'roc_auc',
                               cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                         test_size=grid_search_test_size,
                                                         random_state=random_seed),
                               n_jobs=n_jobs)
            
            clf.fit(self.train_x, self.train_y)
            
        else :
            clf = list(model_grid.keys())[0].set_params(**clf_params)
            clf.fit(self.train_x, self.train_y)
            
        # Add extra RFECV plus redo thetraining (with or without grid search)
        # it will fine tune the number of features and remove collinearity
        if rfe :
            
            if grid_search :
                
                if rfe_model == 'XGBoost' :
                    rfe_clf = XGBClassifier(n_jobs=-2, random_state=random_seed).set_params(**rfe_params)
                elif rfe_model == 'eli5' :
                    rfe_clf = PermutationImportance(clf.best_estimator_, 
                                                    scoring='roc_auc',
                                                    cv=None, 
                                                    random_state=random_seed)
                else :
                    rfe_clf = clf.best_estimator_
                
                rfecv = RFECV(rfe_clf,
                              scoring = 'roc_auc',
                              cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                         test_size=grid_search_test_size,
                                                         random_state=random_seed),
                               step = rfe_step,
                               min_features_to_select = min_rfe)
                              
                self.train_x = rfecv.fit_transform(self.train_x, self.train_y)
                selected_feats = selected_feats[rfecv.support_]
                
                # redo grid search inner test/val split
                clf = GridSearchCV(estimator=list(model_grid.keys())[0], 
                                   param_grid=list(model_grid.values())[0],
                                   scoring = 'roc_auc',
                                   cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                             test_size=grid_search_test_size,
                                                             random_state=random_seed),
                                   n_jobs=n_jobs)
                
                clf.fit(self.train_x, self.train_y)
            
            else :
                if rfe_model == 'XGBoost' :
                    rfe_clf = XGBClassifier(n_jobs=-2, random_state=random_seed).set_params(**rfe_params)
                elif rfe_model == 'eli5' :
                    rfe_clf = PermutationImportance(clf.best_estimator_, 
                                                    scoring='roc_auc',
                                                    cv=None, 
                                                    random_state=random_seed)
                else :
                    rfe_clf = clf

                rfecv = RFECV(rfe_clf, 
                              cv=StratifiedShuffleSplit(n_splits=grid_search_splits, 
                                                         test_size=grid_search_test_size,
                                                         random_state=random_seed),
                               step = rfe_step,
                               min_features_to_select = min_rfe)
                              
                self.train_x = rfecv.fit_transform(self.train_x, self.train_y)
                selected_feats = selected_feats[rfecv.support_]
                
                clf.fit(self.train_x, self.train_y)

            if plot_rfe :
                plt.figure()
                plt.xlabel('Number of features selected')
                plt.ylabel('Cross validation roc auc score')
                plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
                plt.savefig(os.path.join(outdir,'rfe_plot.png'), dpi=600)
                
                
        # output pipeline 
        if rfe : 
            fitted_pipeline = Pipeline([('preprocessing', preprocess),
                                        ('feature_selection', feature_selection),
                                        ('recursive feature elimination', rfecv),
                                        ('classifier', clf)])
        else : 
            fitted_pipeline = Pipeline([('preprocessing', preprocess),
                                        ('feature_selection', feature_selection),
                                        ('classifier', clf)])
        # output selected features 
        self.selected_feats_ = pd.DataFrame(columns = selected_feats, data = self.train_x)
        
        # export the pipeline with joblib as well as final training data
        if outdir != None :
            now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            outpath = os.path.join(outdir,'train_classifier_'+str(list(model_grid.keys())[0])[:re.search('\(', str(list(model_grid.keys())[0])).span()[0]]+'_{}.joblib'.format(now))
            joblib.dump(fitted_pipeline, outpath)
            self.selected_feats_.to_csv(os.path.join(outdir,'training_data_{}.csv'.format(now)), index=False)

            
        print('train_classifier is done', 
              '\n', 
              'the pipeline has been saved in ',
              outpath)
        
        print('Features selected by the pipeline are available through the selected_feats_ attributes')
        

        return fitted_pipeline 




    def train_DNNmodel(self, 
                      feature_names,
                      outdir,
                      random_seed = 0,
                      val_size = 0.2,
                      rfe_search_splits = 5,
                      rfe_search_test_size = 0.25,
                      rfe = True,
                      rfe_step = 1,
                      min_rfe = 1,
                      rfe_model = None,
                      rfe_params = None,
                      plot_rfe = False,
                      kernel_initializer = 'lecun_normal',
                      activation = 'selu',
                      dropout_rate = None,
                      n_output = 1,
                      activation_output = 'sigmoid',
                      loss = 'binary_crossentropy',
                      optimizer = 'nadam',
                      lr = 0.001,
                      metrics = [keras_AUC()],
                      layers = [32,32],
                      weight_l2 = 1,
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
                      full_train = False,
                      preprocess = 'passthrough', 
                      feature_selection = 'passthrough'
                      ) :
        
        """
        Training a DNN classifier
        """
        # future save of DNN parameters and results 
        grid_cols = ['preprocessing', 
                 'feature_selection', 
                 'model', 
                 'hyperparameters',
                 'AUC',
                 'RFE',
                 'features',
                 'num_seed']

        grid_df =  pd.DataFrame(columns=grid_cols, index=[0])

        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        savedir = os.path.join(outdir,'run-{}'.format(now),'model.h5')
        logdir = os.path.join(outdir,'run-{}'.format(now))

        # making sure no models are stored in session()
        K.clear_session()
        
        if callbacks == 'default' :
            callbacks = [ModelCheckpoint(filepath=savedir, monitor=monitor, save_best_only=True, mode=mode),
                         TensorBoard(log_dir=logdir),
                         EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                         ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)]
        else :
            callbacks = callbacks

        # feature_names check
        if isinstance(feature_names, pd.Series) :
            selected_feats = feature_names
        else :
             selected_feats = pd.Series(feature_names)           

        
        # split between train and validation data
        split = StratifiedShuffleSplit(n_splits=1, 
                                       test_size=val_size, 
                                       random_state=random_seed)
        
        for train_index, val_index in split.split(self.train_x, self.train_y):
            X_val = self.train_x.loc[val_index]
            X_train = self.train_x.loc[train_index]
            y_val = self.train_y.loc[val_index]
            y_train = self.train_y.loc[train_index]
        
        X_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)

        # Weighting classes (for imbalanced data if submitted)
        if class_weight :
            class_weight = dict()
            for num_class in range(len(pd.value_counts(y_train))) :
                class_weight[int(num_class)] = (1 / (np.unique(y_train, return_counts=True)[1][num_class]))*(len(y_train))/2.0
        else :
            class_weight = None

                        
        # Fitting the preprocessing pipeline submitted 
        if preprocess == 'passthrough' :
            X_train = X_train.values 
            X_val = X_val.values
        else :
            X_train = preprocess.fit_transform(X_train)
            X_val = preprocess.transform(X_val)    
      
            try :
                preprocess.named_transformers_.num_pipe['polynomial']
            except KeyError :
                pass
            else :
                selected_feats = preprocess.named_transformers_.num_pipe['polynomial'].get_feature_names(selected_feats)
                selected_feats = pd.Series(selected_feats)
              
                
        # Applying feature selection(s) if submitted :
            
        if feature_selection == 'passthrough' :
            pass
            
        else : 
            X_train = feature_selection.fit_transform(X_train, y_train)
            X_val = feature_selection.transform(X_val)
            selected_feats  = selected_feats[feature_selection.support_]
                
        if rfe :
            
            if rfe_model == 'XGBoost' :
                rfe_clf = XGBClassifier(n_jobs=-2, random_state=random_seed).set_params(**rfe_params)
                rfecv = RFECV(rfe_clf,
                              scoring = 'roc_auc',
                              cv=StratifiedShuffleSplit(n_splits=rfe_search_splits, 
                                                     test_size=rfe_search_test_size,
                                                     random_state=random_seed),
                                step = rfe_step,
                                min_features_to_select = min_rfe)
                          
                X_train = rfecv.fit_transform(X_train, y_train)
                X_val = rfecv.transform(X_val)
                selected_feats = selected_feats[rfecv.support_]

                if plot_rfe :
                    plt.figure()
                    plt.xlabel('Number of features selected')
                    plt.ylabel('Cross validation roc auc score')
                    plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
                    plt.savefig(outdir+'rfe_plot_{}.png'.format(rfe_model), dpi=600)

            elif rfe_model == 'DNN' :
                
                K.clear_session()
                
                base_mask = {feat:True for feat in selected_feats}
                
                rfe_dict_auc = dict()
                rfe_dict_mask = dict()
                                
                for count in range(X_train.shape[1], min_rfe, -1) :
                    
                    X_tmp = X_train[:,list(base_mask.values())]
                    X_val_tmp = X_val[:,list(base_mask.values())]
                    
                    clf = make_DNNmodel(input_dim = X_tmp.shape[1],
                           kernel_initializer = kernel_initializer,
                           kernel_seed = random_seed,
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
                        clf.fit(X_tmp, y_train, 
                                validation_data=(X_val_tmp, y_val), 
                                epochs = epochs, 
                                callbacks=[ModelCheckpoint(filepath=os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)), 'model.h5'), monitor=monitor, save_best_only=True, mode=mode),
                                           TensorBoard(os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)))),
                                           EarlyStopping(monitor= 'val_loss', min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                           ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)], 
                                class_weight=class_weight,
                                batch_size=batch_size
                                )
# =============================================================================
#                     else :
#                         training_generator = BalancedBatchGenerator(X_tmp,
#                                                                     y_train,
#                                                                     batch_size=batch_size,
#                                                                     random_state=random_seed)
#                         clf.fit(training_generator, 
#                                 validation_data=(X_val_tmp, y_val), 
#                                 epochs = epochs, 
#                                 callbacks=[ModelCheckpoint(filepath=os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)), 'model.h5'),monitor=monitor, save_best_only=True, mode=mode),
#                                            TensorBoard(os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)))),
#                                            EarlyStopping(monitor= 'val_loss', min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
#                                            ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)], 
#                                 class_weight=class_weight
#                                 )
#                     
# =============================================================================
                    # storing model's auc for future selection of the best subset of features
                    y_val_DNN = clf.predict(X_val_tmp)
                    DNN_fpr, DNN_tpr, DNN_thresholds = roc_curve(y_val, y_val_DNN)
                    DNN_auc = auc(DNN_fpr, DNN_tpr)
                    rfe_dict_auc[str(count)] = round(DNN_auc, 4)
                    rfe_dict_mask[str(count)] = list(base_mask.values()).copy()
                    
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

                    base_score, score_decreases = get_score_importances(score_importance, X_val_tmp, y_val, n_iter=5, random_state = random_seed)
                    feature_importances = list(np.mean(score_decreases, axis=0))
                    feat_to_remove = feature_importances.index(min(feature_importances))
                    key_to_remove = list({i:j for i, j in base_mask.items() if j == True}.keys())[feat_to_remove]
                    base_mask[key_to_remove] = False
                    
                    K.clear_session()
                    print(abs(count-X_train.shape[1]-1), '/', ((X_train.shape[1])-min_rfe), 'rfe done')
                
                # plotting rfe if true
                if plot_rfe :
                    plt.figure()
                    plt.xlabel('Number of features selected')
                    plt.ylabel('Validation roc auc score')
                    plt.plot(range(X_train.shape[1], X_train.shape[1]-len(rfe_dict_auc), -1), list(rfe_dict_auc.values()))
                    plt.savefig(os.path.join(outdir, 'rfe_plot_DNN_{}.png'.format(now)), dpi=600)

                # finding best result                                                                 
                best_key = max(rfe_dict_auc, key=rfe_dict_auc.get)
                print('Best number of feature is ', best_key)
                best_mask = rfe_dict_mask[best_key]
                   
                selected_feats  = selected_feats[best_mask]
                dnnrfe = Mask_selector(best_mask)
                X_train = dnnrfe.fit_transform(X_train)
                X_val = dnnrfe.transform(X_val)


            else :
                raise ValueError 
                print('only XGBoost/DNN is a valid model for RFE at the moment')               


        # Training Final DNN model (except for rfe DNN where each model has been saved already)        
        else : 

            K.clear_session()
            # Looping over each model hyperparameters
            clf = make_DNNmodel(input_dim = X_train.shape[1],
                   kernel_initializer = kernel_initializer,
                   kernel_seed = random_seed,
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
                
                if full_train == False :    
                    history = clf.fit(X_train, y_train, 
                            validation_data=(X_val, y_val), 
                            epochs = epochs, 
                            callbacks=callbacks, 
                            class_weight=class_weight,
                            batch_size=batch_size)
                else :
                    history = clf.fit(self.train_x, self.train_y, 
                            validation_data=None, 
                            epochs = epochs, 
                            callbacks=[TensorBoard(log_dir = logdir)], 
                            class_weight=class_weight,
                            batch_size=batch_size)
                    clf.save(savedir)
                    
# =============================================================================
#             else :
#                 training_generator = BalancedBatchGenerator(X_train, 
#                                                             y_train, 
#                                                             sampler=RandomUnderSampler(random_state=random_seed), 
#                                                             random_state=random_seed)
#                 history = clf.fit(training_generator, 
#                         validation_data=(X_val, y_val), 
#                         epochs = epochs, 
#                         callbacks=callbacks, 
#                         class_weight=class_weight,
#                         batch_size=batch_size)
#     
#     
# =============================================================================
            K.clear_session()
                                                        
        # dnn params
        DNN_params = {
                'kernel_initializer' : kernel_initializer,
                'activation' : activation,
                'dropout_rate' : dropout_rate,
                'n_output' : n_output,
                'activation_output' : activation_output,
                'loss' : loss,
                'optimizer' : optimizer,
                'learning_rate' : lr,
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
        grid_df.loc[0, 'preprocessing'] = \
            str(preprocess)       
        grid_df.loc[0, 'feature_selection'] = \
            str(feature_selection)        

        grid_df.loc[0, 'model'] = \
            str('DNN')
        
        grid_df.loc[0, 'hyperparameters'] = \
            str(DNN_params)

        if (rfe_model != 'DNN') & (full_train == False) :
            grid_df.loc[0, 'val_AUC'] = round(np.max(history.history['val_auc']), 3)
        elif full_train :
            grid_df.loc[0, 'val_AUC'] = 'trained on full train set'            
        else :        
            grid_df.loc[0, 'val_AUC'] = 'see rfe_auc'

        
        if rfe :
            grid_df.loc[0, 'RFE'] = str(rfe_model)
        else :        
            grid_df.loc[0, 'RFE'] = 'No'
        
        grid_df.loc[0, 'features'] = str(selected_feats)

        grid_df.loc[0, 'num_seed'] = int(random_seed)

        if (rfe == True)&(rfe_model == 'DNN') :
            grid_df.loc[0, 'rfe_mask'] = str(rfe_dict_mask)
            grid_df.loc[0, 'rfe_auc'] = str(rfe_dict_auc)
            grid_df.loc[0, 'rfe_best'] = str(best_key)
                
                
        # output sklearn pipeline (keras model is already outputed with callback)
        if rfe_model == 'XGBoost' : 
            fitted_pipeline = Pipeline([('preprocessing', preprocess),
                                        ('feature_selection', feature_selection),
                                        ('recursive feature elimination', rfecv)])

        elif rfe_model == 'DNN' : 
            fitted_pipeline = Pipeline([('preprocessing', preprocess),
                                        ('feature_selection', feature_selection),
                                        ('recursive feature elimination', dnnrfe)])

        else : 
            fitted_pipeline = Pipeline([('preprocessing', preprocess),
                                        ('feature_selection', feature_selection)])
       
        # keep selected features 
        self.train_x = X_train
        self.train_y = y_train
        self.selected_feats_ = pd.DataFrame(columns = selected_feats, data = self.train_x)
        self.validation_data_ = pd.DataFrame(columns = selected_feats, data = X_val)
        self.val_y = y_val
       
        # export the pipeline with joblib, and the final training data
        if outdir != None :
            outpath = os.path.join(outdir, 'train_classifier_DNN_{}.joblib'.format(now))
            joblib.dump(fitted_pipeline, outpath)
            self.selected_feats_.to_csv(os.path.join(outdir, 'training_data_{}.csv'.format(now)), index=False)
            self.train_y.to_csv(os.path.join(outdir, 'target_data_{}.csv'.format(now)), index=False)
            self.validation_data_.to_csv(os.path.join(outdir, 'validation_data_{}.csv'.format(now)), index=False)
            self.val_y.to_csv(os.path.join(outdir, 'target_validation_data_{}.csv'.format(now)), index=False)
            grid_df.to_csv(os.path.join(outdir, 'training_report_{}.csv'.format(now)), index=False)
            
        print('train_classifier is done', 
              '\n', 
              'the pipeline has been saved in ',
              outdir)
        
        print('Features selected by the pipeline are available through the selected_feats_ attributes')
        

        return fitted_pipeline 


    def evaluate_classifier(self, 
                            clf,
                            features_selected,
                            outdir, 
                            title, 
                            keras = False,
                            dpi = 600, 
                            extra_metrics = False,
                            use_shap  = True,                           
                            shap_type = 'kernel',
                            dependence_plot = False) :
        
        """
        evaluate a fitted classification pipeline by outputting its metrics on 
        the test set with a roc_plot (also displaying (balanced) accuracy, 
        precision and recall values
        It also outputs shap visualizations
        """
        
        # output metrics on the test set on the roc curve
        if keras :
            y_test_predict_p = clf.predict(self.test_x)                      
            fpr, tpr, thresholds = roc_curve(self.test_y, y_test_predict_p)
        else :
            y_test_predict_p = clf.predict_proba(self.test_x)                      
            fpr, tpr, thresholds = roc_curve(self.test_y, y_test_predict_p[:,1])
           
        roc_auc = auc(fpr, tpr)
        youden = (1-fpr)+tpr-1
        best_threshold = np.where(youden == np.max(youden))[0][0]
        default_threshold = np.argmin(np.abs(thresholds-.5))
        severe_threshold = np.argmin(np.abs(thresholds-.9))
        
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[severe_threshold], tpr[severe_threshold], 'o', color = 'orange')
        plt.plot(fpr[best_threshold], tpr[best_threshold], 'o', color = 'red')
        plt.plot(fpr[default_threshold], tpr[default_threshold], 'o', color = 'blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC - '+title)
        plt.legend(loc="lower right")
        tmp_sub=0.01
        plt.text(fpr[severe_threshold]+.01, tpr[severe_threshold]-tmp_sub, 'Severe: ' + str(round(thresholds[severe_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[severe_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[severe_threshold]), 1))) + ')')
        plt.text(fpr[best_threshold]+.01, tpr[best_threshold]+0, 'Optimal: ' + str(round(thresholds[best_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[best_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[best_threshold]), 1))) + ')')
        plt.text(fpr[default_threshold]+.01, tpr[default_threshold]-0, 'Default: ' + str(round(thresholds[default_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[default_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[default_threshold]), 1))) + ')')
        if extra_metrics :
            y_test_predict = clf.predict(self.test_x)
            clf_acc = accuracy_score(self.test_y, y_test_predict)
            clf_bal_acc = balanced_accuracy_score(self.test_y, y_test_predict)
            clf_pre = precision_score(self.test_y, y_test_predict)
            clf_rec = recall_score(self.test_y, y_test_predict)        
            plt.text(0.7, 0.1, 'Accuracy: '+str(round(clf_acc, 3))+'\n'+
                'Balanced Accuracy: '+str(round(clf_bal_acc, 3))+'\n'+
                'Precision: '+str(round(clf_pre, 3))+'\n'+
                'Recall: '+str(round(clf_rec, 3)))
        
        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        outpath = os.path.join(outdir,'evaluate_classifier_ROC_plot_{}.png'.format(now))

        plt.savefig(outpath, dpi=dpi)
        
        
        # now outputting shap plots
        if use_shap :            
            if shap_type == 'kernel' :
                
                if keras :
                    X_train_summary = shap.kmeans(self.train_x, 30)                    
                    shap_values = shap.KernelExplainer(clf.predict, X_train_summary).shap_values(self.train_x)
                    # making summary_plot
                    summaryplot = plt.figure()
                    shap.summary_plot(shap_values[0], features_selected.reset_index(drop=True))
                    summaryplot.savefig(os.path.join(outdir,'evaluate_classifier_summary_plot_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)

                else :
                    X_train_summary = shap.kmeans(self.train_x, 30)
                    shap_values = shap.KernelExplainer(clf.named_steps.classifier.predict_proba, X_train_summary).shap_values(self.train_x)
                    # making summary_plot
                    summaryplot = plt.figure()
                    shap.summary_plot(shap_values[1], features_selected)
                    summaryplot.savefig(os.path.join(outdir,'evaluate_classifier_summary_plot_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)

            
# =============================================================================
#             if shap_type == 'tree' :
#                 shap_values = shap.TreeExplainer(clf.named_steps.classifier.predict_proba, self.train_x).shap_values(self.train_x)
#                 
#             if shap_type == 'deep' :
#                 shap_values = shap.DeepExplainer(clf.predict, self.train_x).shap_values(self.train_x)
#                 summaryplot = plt.figure()
#                 shap.summary_plot(shap_values, features_selected)
#                 summaryplot.savefig(outdir+'evaluate_classifier_summary_plot_{}.png'.format(now), bbox_inches='tight', dpi=dpi)
# 
# =============================================================================


            # making dependence_plot
            if dependence_plot :
                if keras :
                    for feat in features_selected :
                        dependenceplot = plt.figure()
                        shap.dependence_plot(list(features_selected.columns).index(feat), shap_values[0], self.train_x, feature_names=features_selected.columns)
                        dependenceplot.savefig(os.path.join(outdir, 'evaluate_classifier_dependance_plot_'+str(feat)+'_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)
                else :
                    for feat in features_selected :
                        dependenceplot = plt.figure()
                        shap.dependence_plot(list(features_selected.columns).index(feat), shap_values[1], self.train_x, feature_names=features_selected.columns)
                        dependenceplot.savefig(os.path.join(outdir, 'evaluate_classifier_dependance_plot_'+str(feat)+'_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)
                
        print('evaluation plots have been written in ', outpath)
        
        
        return None



