# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:14:19 2020

@author: Xavier Dieu
"""

# =============================================================================
# Custom Functions for Machine Learning
# 
# =============================================================================

# imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.feature_selection import RFECV 
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc, roc_curve
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.keras import BalancedBatchGenerator
import joblib
import shap
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.initializers import he_normal, glorot_uniform, lecun_normal
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC as keras_AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers import Nadam

# Functions 

def train_test_stratified_splitter(df, test_size, random_state, target,
                                   verbose=1) :
    
    """
    Execute a stratifiedshufflesplit on a target variable 
    Calls sklearn class StratifiedShuffleSplit
    
    df = pandas dataframe
    test_size = number(int) or proportion(float) of test samples
    random_state = random seed (int)
    target = column name of a pandas dataframe (the "y")
    verbose (default=1) = prints summary stats of the split
    
    returns 2 df (the train and the test)
    """
    
    split = StratifiedShuffleSplit(n_splits=1, 
                                   test_size=test_size, 
                                   random_state=random_state)
    
    for train_index, test_index in split.split(df, df[target]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
  
    if verbose > 0 :
        print('Stratified sampling resulted in ',
              len(train_set), 'train +', 
              len(test_set), 
              'test samples')        
        print('')
        
        # let's check if the stratified sampling did work
        print('stratified sampling results in train set : ', 
              df[target].value_counts()/len(df),
              '\n',
              '\n'
              'stratified sampling results in test set : ',
              test_set[target].value_counts()/len(test_set))
        print('')

    return train_set, test_set



def plot_df_nan(df) :
    
    """
    Plot the number of features given their percentage of missing value
    using a pandas hist() method
    """
    
    NaNPerCols = (round(df.isna().sum()/df.shape[0], 3)).to_dict()
    pd.DataFrame.from_dict(NaNPerCols, orient='index').hist()
    plt.title('Number of features given their percentage of missing value')
    
    return plt.show()



def nan_filter(df, threshold) :
    
    """
    filter the the desired % of missing values (arg 'threshold')
    return a filtered df where columns' nan % < to the threshold are kept
    also return a list of the dropped features
    """

    NaNPerCols = (round(df.isna().sum()/df.shape[0], 3)).to_dict()
    NaNPerCols_filtered = {i:j for i, j in NaNPerCols.items() if j < threshold}
    NaNPerCols_droped = {i:j for i, j in NaNPerCols.items() if j > threshold}
    
    print('features droped: % nans \n', NaNPerCols_droped)
    print('')
    
    return df[list(NaNPerCols_filtered.keys())], NaNPerCols_droped



def preprocessor(feats, num_feat, cat_feat, pipelines=[], random=42, n_components=20, custom_pipelines=None) :
    
    """
    Function that returns the desired preprocessing pipelines
    
    pipelines args (type the corresponding integer(s) in a list):
        1 'preprocess_pipeline1 (Sscale, BRnum_imp, ETcat_imp)'
        2 'preprocess_pipeline2 (Sscale, ETnum_imp, ETcat_imp)'
        3 'preprocess_pipeline3 (Qscale, BRnum_imp, ETcat_imp)'
        4 'preprocess_pipeline4 (Qscale, ETnum_imp, ETcat_imp)'
        5 'preprocess_pipeline5 (Rscale, BRnum_imp, ETcat_imp)'
        6 'preprocess_pipeline6 (Rscale, ETnum_imp, ETcat_imp)'
        7 'preprocess_pipeline7 (Sscale, BRnum_imp, ETcat_imp, pca)'
        8 'preprocess_pipeline8 (Sscale, ETnum_imp, ETcat_imp, pca)'
        9 'preprocess_pipeline9 (Qscale, BRnum_imp, ETcat_imp, pca)'
        10 'preprocess_pipeline10 (Qscale, ETnum_imp, ETcat_imp, pca)'
        11 'preprocess_pipeline11 (Rscale, BRnum_imp, ETcat_imp, pca)'
        12 'preprocess_pipeline12 (Rscale, ETnum_imp, ETcat_imp, pca)'
        
    num_feat = list of the numerical features names
    
    cat_feat = list of the categorical features names
    
    random = random seed to feed the pipelines
    
    n_components = n_components to keep when pca is used (on num_feat only)
    
    custom_pipelines = user made custom preprocessing pipelines 
                       (must be a dict, key(s) is a string name and value(s))
                       the pipeline)
    """
    
    # Imputers
    num_imputer1 = BayesianRidge(n_iter=300, fit_intercept = False)
    num_imputer2 = ExtraTreesRegressor(random_state=random)
    num_imputer3 = SVR()
    cat_imputer = ExtraTreesClassifier(random_state=random)
    
    
    # pipeline for categorical features
    cat_pipeline = Pipeline([('imputer', IterativeImputer(cat_imputer, random_state=random))])
    
    # pipeline for numerical features
    num_pipeline1 = Pipeline([('prescale', StandardScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', StandardScaler())])
        
    num_pipeline2 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', StandardScaler())])

    num_pipeline3 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', QuantileTransformer())])
    
    num_pipeline4 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', QuantileTransformer())])
    
    num_pipeline5 = Pipeline([('prescale', RobustScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', RobustScaler())])
    
    num_pipeline6 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', RobustScaler())])
    
    num_pipeline7 = Pipeline([('prescale', StandardScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', StandardScaler()),
                             ('pca', PCA(n_components=n_components))])
        
    num_pipeline8 = Pipeline([('prescale', StandardScaler()),
                             ('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', StandardScaler()),
                             ('pca', PCA(n_components=n_components))])

    num_pipeline9 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', QuantileTransformer()),
                             ('pca', PCA(n_components=n_components))])
    
    num_pipeline10 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', QuantileTransformer()),
                             ('pca', PCA(n_components=n_components))])
    
    num_pipeline11 = Pipeline([('prescale', RobustScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', RobustScaler()),
                             ('pca', PCA(n_components=n_components))])
    
    num_pipeline12 = Pipeline([('prescale', RobustScaler()),
                             ('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', RobustScaler()),
                             ('pca', PCA(n_components=n_components))])
    
    num_pipeline13 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random))])

    num_pipeline14 = Pipeline([('prescale', StandardScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', StandardScaler()),
                             ('polynomial', PolynomialFeatures(degree=2))])
        
    num_pipeline15 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('prescale', StandardScaler()),
                             ('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
                             ('postscale', StandardScaler())])

    num_pipeline16 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', QuantileTransformer()),
                             ('polynomial', PolynomialFeatures(degree=2))])
    
    num_pipeline17 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', QuantileTransformer()),
                             ('polynomial', PolynomialFeatures(degree=2))])
    
    num_pipeline18 = Pipeline([('prescale', RobustScaler()),
                             ('imputer', IterativeImputer(num_imputer1, random_state=random)),
                             ('postscale', RobustScaler()),
                             ('polynomial', PolynomialFeatures(degree=2))])
    
    num_pipeline19 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                               ('prescale', RobustScaler()),                             
                               ('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
                               ('postscale', RobustScaler())])

    num_pipeline20 = Pipeline([('prescale', QuantileTransformer()),
                             ('imputer', IterativeImputer(num_imputer3, random_state=random)),
                             ('postscale', QuantileTransformer())])

    num_pipeline21 = Pipeline([('imputer', IterativeImputer(num_imputer2, random_state=random)),
                             ('postscale', QuantileTransformer(output_distribution='normal'))])

    num_pipeline22 = Pipeline([('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
                             ('postscale', StandardScaler())])

    num_pipeline23 = Pipeline([('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
                               ('postscale', StandardScaler()),
                               ('pca', PCA()),
                             ('postscale2', StandardScaler())])

    
    # Preprocessing pipelines
    preprocess_pipeline1 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline1, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline2 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline2, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline3 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline3, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
    
    preprocess_pipeline4 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline4, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline5 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline5, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline6 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline6, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
 
    preprocess_pipeline7 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline7, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline8 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline8, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline9 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline9, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
    
    preprocess_pipeline10 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline10, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
       
    preprocess_pipeline11 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline11, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')
        
    preprocess_pipeline12 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline12, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline13 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline13, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline14 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline14, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline15 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline15, feats)
            ], remainder='drop')

    preprocess_pipeline16 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline16, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline17 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline17, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline18 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline18, feats)
            ], remainder='drop')

    preprocess_pipeline19 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline19, feats)
            ], remainder='drop')

    preprocess_pipeline20 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline20, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline21 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline21, num_feat),
            ('cat_pipe', cat_pipeline, cat_feat)
            ], remainder='drop')

    preprocess_pipeline22 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline13, feats),
            ], remainder='drop')

    preprocess_pipeline23 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline2, feats),
            ], remainder='drop')

    preprocess_pipeline24 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline4, feats),
            ], remainder='drop')

    preprocess_pipeline25 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline6, feats),
            ], remainder='drop')

    preprocess_pipeline26 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline22, feats),
            ], remainder='drop')

    preprocess_pipeline27 = ColumnTransformer(transformers= [
            ('num_pipe', num_pipeline23, feats),
            ], remainder='drop')

    
    # storing all pipelines in a dict with a descriptive key
    preprocess_pipelines = dict()
    
    try :
        pipelines.index(1) 
    except ValueError :
        pass
    else :
        preprocess_pipelines['preprocess_pipeline1 (Sscale, BRnum_imp, ETcat_imp)'] = \
            preprocess_pipeline1

    try :
        pipelines.index(2) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline2 (Sscale, ETnum_imp, ETcat_imp)'] = \
            preprocess_pipeline2
    
    try :
        pipelines.index(3) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline3 (Qscale, BRnum_imp, ETcat_imp)'] = \
            preprocess_pipeline3

    try :
        pipelines.index(4) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline4 (Qscale, ETnum_imp, ETcat_imp)'] = \
            preprocess_pipeline4

    try :
        pipelines.index(5) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline5 (Rscale, BRnum_imp, ETcat_imp)'] = \
            preprocess_pipeline5

    try :
        pipelines.index(6) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline6 (Rscale, ETnum_imp, ETcat_imp)'] = \
            preprocess_pipeline6

    try :
        pipelines.index(7) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline7 (Sscale, BRnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline7

    try :
        pipelines.index(8) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline8 (Sscale, ETnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline8

    try :
        pipelines.index(9) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline9 (Qscale, BRnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline9

    try :
        pipelines.index(10) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline10 (Qscale, ETnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline10

    try :
        pipelines.index(11) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline11 (Rscale, BRnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline11

    try :
        pipelines.index(12) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline12 (Rscale, ETnum_imp, ETcat_imp, pca)'] = \
            preprocess_pipeline12

    try :
        pipelines.index(13) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline13 (ETnum_imp, ETcat_imp)'] = \
            preprocess_pipeline13

    try :
        pipelines.index(14) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline14 (ETnum_imp, ETcat_imp, polynomial feats)'] = \
            preprocess_pipeline14

    try :
        pipelines.index(15) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline15 (ETall_imp, polynomial feats)'] = \
            preprocess_pipeline15

    try :
        pipelines.index(16) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline16 (ETnum_imp, ETcat_imp, polynomial feats)'] = \
            preprocess_pipeline16

    try :
        pipelines.index(17) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline17 (ETnum_imp, ETcat_imp, polynomial feats)'] = \
            preprocess_pipeline17

    try :
        pipelines.index(18) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline18 (ETnum_imp, ETcat_imp, polynomial feats)'] = \
            preprocess_pipeline18

    try :
        pipelines.index(19) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline19 (ETall_imp, polynomial feats)'] = \
            preprocess_pipeline19


    try :
        pipelines.index(20) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline20 (Qscale, SVRnum_imp, ETcat_imp)'] = \
            preprocess_pipeline20

    try :
        pipelines.index(21) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline21 (Qscale(norm), ETnum_imp, ETcat_imp'] = \
            preprocess_pipeline21

    try :
        pipelines.index(22) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline22 (ETall_imp)'] = \
            preprocess_pipeline22

    try :
        pipelines.index(23) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline23 (ETall_imp Sscale)'] = \
            preprocess_pipeline23

    try :
        pipelines.index(24) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline24 (ETall_imp Qscale)'] = \
            preprocess_pipeline24

    try :
        pipelines.index(25) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline21 (ETall_imp Rscale)'] = \
            preprocess_pipeline25

    try :
        pipelines.index(26) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline26 (polynom, Sscale)'] = \
            preprocess_pipeline26

    try :
        pipelines.index(27) 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['preprocess_pipeline27 (polynom, pca, Sscale)'] = \
            preprocess_pipeline27

    try :
        pipelines.index('passthrough') 
    except ValueError :
        pass
    else :    
        preprocess_pipelines['passthrough'] = \
            'passthrough'

    # add custom user pipelines
    if custom_pipelines != None :
        for key in custom_pipelines.keys() :
            preprocess_pipelines[key] = custom_pipelines[key]
    
    
    return preprocess_pipelines



def feature_selector(feature_selection = True,
                     passthrough = True,
                     n_estimators='auto',
                     max_depth = 5, 
                     perc = 90,
                     max_iter = 100,
                     random = 42, 
                     custom_selector=None) :
    
    """
    Function to output a feature_selection pipeline 
    feature selection with borutaPy and ExtraTrees
    passthrough to add an empty pipeline
    custom_pipelines = user made custom preprocessing pipelines 
                   (must be a dict, key(s) is a string name and value(s))
                   the pipeline/model with fit_transform method)
    """
    
    ET_model = ExtraTreesClassifier(n_jobs=-1, max_depth=max_depth, random_state=random) # 
    ET_selector = BorutaPy(ET_model, 
                                n_estimators=n_estimators, 
                                perc=perc, 
                                verbose=0, 
                                max_iter=max_iter, 
                                random_state=random)              
            
    feat_selec_pipelines = dict()
    
    if feature_selection == True :    
        feat_selec_pipelines['feature_selection (BorutaPy)'] = ET_selector
    
    if passthrough == True :    
        feat_selec_pipelines['no_feature_selection'] = 'passthrough'
    
    if custom_selector != None :
        for key in custom_selector.keys() :
            feat_selec_pipelines[key] = custom_selector[key]
    
    return feat_selec_pipelines



def stability_selector(X, 
                     y, 
                     feats,
                     feature_selector,
                     stability_iteration = 3,
                     stability_size = 0.75,
                     keep_count = 3) :
    
    """
    Function to output a feature_selection pipeline 
    feature selection done with a feature_selector function
    
    """
    
    # storing the number of times a feature will be selected
    FS = pd.DataFrame(columns=['Counts'], data=np.zeros((X.shape[1],1)), index=feats)
    #recreating the original df with column names
    data = pd.DataFrame(columns=feats, data=X)
    
    for i in range(stability_iteration) :
        
        # make use of SSS for selecting a random stratified subset of our samples 
        split = StratifiedShuffleSplit(n_splits=1, test_size=(1-stability_size), random_state=i)
        for stability_index, _ in split.split(data, y):
            stability_set = data.loc[stability_index, :]
    
        feature_selector.fit(stability_set, y)
        feats_selected = data.loc[:,feature_selector.support_]
        for feature in FS.index :
            if feature in list(feats_selected.columns) :
                FS.loc[feature,'Counts'] += 1
                    
            
    feat_selec_mask = list(FS['Counts']>=keep_count)
        
    return feat_selec_mask



def make_DNNclassifier(input_dim,
                       kernel_initializer = 'glorot_uniform',
                       kernel_seed = 0,
                       activation = 'relu',
                       dropout_rate = None,
                       n_output = 1,
                       activation_output = 'sigmoid',
                       loss = 'binary_crossentropy',
                       optimizer = 'nadam',
                       lr = 0.001,
                       metrics = [keras_AUC()],
                       layers = [32,32],
                       weight_l2 = None,
                       weight_l2_output = None,
                       batch_norm = None) :
    
    """
    create a Keras DNN classifier 
    """
    
    model = Sequential()
    
    # adding dense architecture of the DNN

    if kernel_initializer == 'he_normal' :
        kernel_initializer = he_normal(seed=kernel_seed)
    if kernel_initializer == 'glorot_uniform' :
        kernel_initializer = glorot_uniform(seed=kernel_seed)
    if kernel_initializer == 'lecun_normal' :
        kernel_initializer = lecun_normal(seed=kernel_seed)

    if weight_l2 == None :
        kernel_regularizer = None
    else :
        kernel_regularizer = l2(weight_l2)
    if weight_l2_output == None :
        kernel_regularizer_output = None
    else :
        kernel_regularizer_output = l2(weight_l2_output)
    
    # Adding layers :
    first_layer = True
    
    for neurons in layers :
    
        if first_layer :                        
            model.add(Dense(neurons,
                            input_dim = input_dim,
                            kernel_initializer = kernel_initializer,
                            kernel_regularizer = kernel_regularizer,
                                   ))
            
            if batch_norm != None :
                model.add(BatchNormalization())
            
            model.add(Activation(activation))
                
            if dropout_rate != None :
                if activation == 'selu' :
                    model.add(AlphaDropout(dropout_rate, seed=kernel_seed))
                else :
                    model.add(Dropout(dropout_rate, seed=kernel_seed))
            
            first_layer = False
            
        else :
            model.add(Dense(neurons,
                            kernel_initializer = kernel_initializer,
                            kernel_regularizer = kernel_regularizer,
                                   ))
            
            if batch_norm != None :
                model.add(BatchNormalization())
            
            model.add(Activation(activation))
                
            if dropout_rate != None :
                if activation == 'selu' :
                    model.add(AlphaDropout(dropout_rate, seed=kernel_seed))
                else :
                    model.add(Dropout(dropout_rate, seed=kernel_seed))

    # output neuron(s)
    model.add(Dense(n_output,
                    activation = activation_output, 
                    kernel_regularizer = kernel_regularizer_output))

    if optimizer == 'nadam' :
        optimizer = Nadam(lr)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    
    return model


    

def classifier_generator(models, random, custom_models = None,
                     strategy=['stratified', 'most_frequent'],
                     max_depth=[5,10, None],
                     min_samples_split=[2, 5, 8],
                     min_samples_leaf=[1, 2, 3, 5],
                     n_estimators=[20,100, 1000],
                     class_weight=[None, 'balanced'],
                     C = [0.1,0.5, 1, 2, 5],
                     n_neighbors = [3,5,10],
                     weights = ['uniform', 'distance'],
                     n_estimators_xgb=[3, 4, 5, 10, 50, 100],
                     learning_rate_xgb = [0.01, 0.1, 0.3],
                     gamma_xgb = [0, 1, 5, 10],
                     max_depth_xgb = [1, 2, 3, 4, 10, 20],
                     min_child_weight = [0.5, 1, 2],
                     lambda_xgb = [1],
                     alpha_xgb = [0],
                     subsample = [1],
                     colsample_bytree = [1],
                     colsample_bynode = [1],
                     scale_pos_weight = [1],
                     loss = ['hinge', 'log', 'modified_huber'],
                     penalty = ['l2', 'elasticnet'],
                     alpha = [0.0001, 0.001],
                     l1_ratio = [0, 0.5, 0.8, 0.9, 1],
                     max_iter = [1000],
                     tol = [0.001],
                     n_jobs=-2,
                     ) : 
    
    """
    Function to return a dict of the different models we want to try 
    as well as hyperparameters we want to tweek 
    models = list of the models we want to add
    choices are Dummy, RandomForest, ExtraTrees, DecisionTree, SVM, KNN 
    We can add custom_models in the form of a dict with key being a model and
    value being the hyperparameters to test
    """
    
    models_grid = dict()
    
    # adding every model and hyperparameters we want to try
    if 'Dummy' in models :
        models_grid[DummyClassifier(random_state=random)] = {
                     'strategy': strategy,
                     }
    
    if 'RandomForest' in models :
        models_grid[RandomForestClassifier(random_state=random, n_jobs=n_jobs)] = {
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split, 
                     'min_samples_leaf': min_samples_leaf, 
                     'n_estimators': n_estimators , 
                     'class_weight': class_weight 
                     }
    
    if 'ExtraTrees' in models :
        models_grid[ExtraTreesClassifier(random_state=random, n_jobs=n_jobs)] = {
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split, 
                     'min_samples_leaf': min_samples_leaf, 
                     'n_estimators': n_estimators , 
                     'class_weight': class_weight 
                     }
    
    if 'DecisionTree' in models :
        models_grid[DecisionTreeClassifier(random_state=random)] = {
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split, 
                     'min_samples_leaf': min_samples_leaf, 
                     'class_weight': class_weight 
                     }

    if 'SVM' in models :   
        models_grid[SVC(kernel='rbf', probability=True, random_state=random)] = {
                     'C': C,
                     'class_weight': class_weight
                     }
    
    if 'KNN' in models :
        models_grid[KNeighborsClassifier(n_jobs=n_jobs)] = {
                     'n_neighbors': n_neighbors,
                     'weights': weights, 
                     }

    if 'XGBoost' in models :
        models_grid[XGBClassifier(n_jobs=n_jobs, random_state=random)] = {
                      'n_estimators' : n_estimators_xgb,
                      'learning_rate' : learning_rate_xgb,
                      'gamma' : gamma_xgb,
                     'max_depth': max_depth_xgb,
                     'min_child_weight': min_child_weight,
                     'lambda' : lambda_xgb,
                     'alpha' : alpha_xgb,
                     'subsample' : subsample,
                     'colsample_bytree' : colsample_bytree,
                     'colsample_bynode' : colsample_bynode,
                     'scale_pos_weight' : scale_pos_weight
                     }

    if 'LinearSVM' in models :
        models_grid[SVC(kernel='linear', probability=True, random_state=random)] = {
                     'C': C,
                     'class_weight': class_weight
                     }

    if 'SGDclassifier' in models :
        models_grid[SGDClassifier(random_state=random, n_jobs=n_jobs)] = {
                     'loss': loss,
                     'penalty': penalty,
                     'alpha': alpha,
                     'l1_ratio': l1_ratio,
                     'max_iter': max_iter,
                     'tol': tol,
                     'class_weight': class_weight
                     }

    if custom_models != None :
        for key in custom_models.keys() :
            models_grid[key] = custom_models[key]

    
    return models_grid




def plot_confusion_matrix(y_true, y_pred, *, labels=None,
                          sample_weight=None, normalize=None,
                          include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None,
                          cmap='viridis', ax=None):
    
    import sklearn.metrics as skmet

    cm = skmet.confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    display_labels = labels

    disp = skmet.ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
                     values_format=values_format)




# Classes

class DNNRFE(BaseEstimator, TransformerMixin) :
    """
    custom transformer for DNNrfe to use in train_DNNclassifier
    """
    
    def __init__(self, mask) :
        self.support_ = mask
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        return X[:, self.support_]



class CARTselector(BaseEstimator, TransformerMixin) :

    """
    custom transformer for ranking and selecting features through a CART
    """
    
    def __init__(self,  mode='class', seed=0, outdir=None) :
        self.mode = mode
        self.seed = seed
        self.outdir = outdir
        self.support_ = None
        self.result_df = None
        self.poly_feat = None
        
    def fit(self, X, y, feat_names, cv = 10, poly = True, interaction_only = False) :
        
        # preprocessing with Polynomial features to add possible features extraction
        if poly :
            Poly_feat = PolynomialFeatures(degree = 2, interaction_only = interaction_only, include_bias = False)
            X_p = Poly_feat.fit_transform(X)
            feat_names = Poly_feat.get_feature_names(feat_names)
            X_p = pd.DataFrame(data=X_p, columns=feat_names)
            self.poly_feat = Poly_feat
        else :
            X_p = X
            
        # adding all combinations of 2 features 
        comb_feats = list(combinations(feat_names, 2))
        feat_names.extend(comb_feats)
        
        # instantiating a dataframe to store results
        if self.mode == 'class' :
            score_name = 'AUC'
        elif self.mode == 'reg' :
            score_name = 'RMSE'
        else :
            raise ValueError
            print('non valid mode argument')
        self.result_df = pd.DataFrame(columns=[score_name], index=feat_names)
        
        # fitting the cart on each features or combination of features
        for feat in self.result_df.index : 
            X_tmp = X_p.loc[:,feat]
            if isinstance(feat, str) :
                X_tmp = X_tmp.values.reshape(-1, 1)
            CART = DecisionTreeClassifier(random_state=self.seed, max_depth=3)
            if self.mode == 'class' :
                score = cross_val_score(CART, X_tmp, y, scoring='roc_auc', cv=cv, n_jobs=-2)
            
            self.result_df.loc[feat, score_name] = np.median(score)
        
        # exports of results 
        if self.outdir != None :
            now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            outpath = os.path.join(self.outdir,'CART_selector_{}.xlsx'.format(now))
            self.result_df.to_excel(outpath)  
    
        return self.result_df
    
    def transform(self, X, feat_names, threshold=20, feat_list = None) :
        
        if feat_list == None :
            if self.mode == 'class' :
                self.result_df = self.result_df.sort_values('AUC', ascending = False)
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
    

    def search_classifiers(self, 
                      preprocess_pipelines, 
                      feature_selection_pipelines,
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



    def search_DNNclassifiers(self, 
                      preprocess_pipelines, 
                      feature_selection_pipelines,
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
                      mode = 'min'
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
                                                                        
                                                                        clf = make_DNNclassifier(input_dim = X_tmp.shape[1],
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
                                                                        else :
                                                                            training_generator = BalancedBatchGenerator(X_tmp,
                                                                                                                        y_nes_train,
                                                                                                                        batch_size=batch_size,
                                                                                                                        random_state=i)
                                                                            clf.fit(training_generator, 
                                                                                    validation_data=(X_val_tmp, y_nes_val), 
                                                                                    epochs = epochs, 
                                                                                    callbacks=callbacks, 
                                                                                    class_weight=class_weight
                                                                                    )
                                                                        
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

                                                                    clf = make_DNNclassifier(input_dim = X_nes_train.shape[1],
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
                                                                    else :
                                                                        training_generator = BalancedBatchGenerator(X_nes_train,
                                                                                                                    y_nes_train,
                                                                                                                    batch_size=batch_size,
                                                                                                                    random_state=i)
                                                                        clf.fit(training_generator, 
                                                                                validation_data=(X_nes_val, y_nes_val), 
                                                                                epochs = epochs, 
                                                                                callbacks=callbacks, 
                                                                                class_weight=class_weight
                                                                                )

                                                                        
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




    def search_additive_DNNclassifiers(self, 
                      preprocessing, 
                      feature_selection,
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
                      mode = 'min'
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
        
                    clf = make_DNNclassifier(input_dim = X_nes_train.shape[1],
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
                    else :
                        training_generator = BalancedBatchGenerator(X_nes_train,
                                                                    y_nes_train,
                                                                    batch_size=batch_size,
                                                                    random_state=seed)
                        clf.fit(training_generator, 
                                validation_data=(X_nes_val, y_nes_val), 
                                epochs = epochs, 
                                callbacks=callbacks, 
                                class_weight=class_weight
                                )
                            
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
                      preprocess, 
                      feature_selection,
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
                      n_jobs = -2
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




    def train_DNNclassifier(self, 
                      preprocess, 
                      feature_selection,
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
                      full_train = False
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
                    
                    clf = make_DNNclassifier(input_dim = X_tmp.shape[1],
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
                    else :
                        training_generator = BalancedBatchGenerator(X_tmp,
                                                                    y_train,
                                                                    batch_size=batch_size,
                                                                    random_state=random_seed)
                        clf.fit(training_generator, 
                                validation_data=(X_val_tmp, y_val), 
                                epochs = epochs, 
                                callbacks=[ModelCheckpoint(filepath=os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)), 'model.h5'),monitor=monitor, save_best_only=True, mode=mode),
                                           TensorBoard(os.path.join(outdir, 'run-{}'.format(now), 'rfe_{}'.format(str(count)))),
                                           EarlyStopping(monitor= 'val_loss', min_delta=min_delta, patience=patience, verbose=1, mode=mode, restore_best_weights=True),
                                           ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr, verbose=1, mode=mode, min_delta=min_delta, min_lr=0)], 
                                class_weight=class_weight
                                )
                    
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
                dnnrfe = DNNRFE(best_mask)
                X_train = dnnrfe.fit_transform(X_train)
                X_val = dnnrfe.transform(X_val)


            else :
                raise ValueError 
                print('only XGBoost/DNN is a valid model for RFE at the moment')               


        # Training Final DNN model (except for rfe DNN where each model has been saved already)        
        else : 

            K.clear_session()
            # Looping over each model hyperparameters
            clf = make_DNNclassifier(input_dim = X_train.shape[1],
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
                    
            else :
                training_generator = BalancedBatchGenerator(X_train, 
                                                            y_train, 
                                                            sampler=RandomUnderSampler(random_state=random_seed), 
                                                            random_state=random_seed)
                history = clf.fit(training_generator, 
                        validation_data=(X_val, y_val), 
                        epochs = epochs, 
                        callbacks=callbacks, 
                        class_weight=class_weight,
                        batch_size=batch_size)
    
    
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




class StackingPipeline :
    
    """
    Easy Machine Learning stacking pipeline class
    """
    
    
    def __init__(self, train_data, train_target, test_data, test_target, features) :
        self.train_x = train_data
        self.test_x = test_data
        self.train_y = train_target
        self.test_y = test_target
        if isinstance(features, pd.Series) :
            self.features = features
        else :
             self.features = pd.Series(features)           
        self.selected_feats_ = None
        self.stacking_results_ = None
        self.blender_train_data = np.array([])


    def train_base_models(self, preprocessing, feature_selection, base_models, outdir, cv = 20, seed=0, rfe=True) :
        
        """
        Takes x base models and train them in cross validation 
        for future training of the blender
        parameters :
            base_models = dict of models (string name) and hyperparameters(as dict)
        """
        
        first = True            
        
        # iterate through each base models of the stacking
        for model in base_models :
                        
            if model != 'DNN' :
                
                # scikit learn training of scikit learn models
                if model == 'XGBoost' :
                    clf = XGBClassifier().set_params(**base_models[model])
                elif model == 'ExtraTrees' :
                    clf = ExtraTreesClassifier().set_params(**base_models[model])
                elif model == 'RandomForest' :
                    clf = RandomForestClassifier().set_params(**base_models[model])
                elif model == 'LinearSVM' :
                    clf = SVC(kernel='linear').set_params(**base_models[model])
                elif model == 'SVM' :
                    clf = SVC(kernel='rbf').set_params(**base_models[model])
                else :
                    raise ValueError
                    print('the model is not available')
                
                # creating the model_pipeline
                pipe = list()
                if preprocessing != None : 
                    pipe.append(('preprocessing', preprocessing))
                if feature_selection != None : 
                    pipe.append(('feature_selection', feature_selection))
                if rfe : 
                    pipe.append(('rfe', RFECV(clf, 
                                              step=step, 
                                              min_features_to_select=min_rfe, 
                                              cv=StratifiedKFold(n_splits=cv, random_state=seed), 
                                              scoring='roc_auc', 
                                              n_jobs=-2)))
                pipe.append(('model', clf))
                 
                model_pipeline = Pipeline(pipe)
                
                # cross validate predictions to have predict proba on the full train set
                model_proba = cross_val_predict(model_pipeline, 
                                                self.train_x, 
                                                self.train_y,
                                                cv=StratifiedKFold(n_splits=cv, random_state=seed), 
                                                n_jobs = -2,
                                                method = 'predict_proba')
                
                if first :
                    self.blender_train_data = model_proba[1].reshape((2,1))
                else :
                    self.blender_train_data = np.hstack((self.blender_train_data, model_proba[1]))
                
                # output model after full train set training
                model_pipeline.fit(self.train_x, self.train_y)
                
                
                
            elif model == 'DNN' :
                
                clf = make_DNNclassifier()                
        
        
            else :        
                raise ValueError
                print('the model is not available')
        
            first = False

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

