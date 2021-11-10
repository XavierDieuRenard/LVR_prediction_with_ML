# -*- coding: utf-8 -*-
"""
author : <xavier.dieu@chu-angers.fr>

Code to do the analysis described in the "POST-INFARCT CARDIAC REMODELING 
PREDICTIONS WITH MACHINE LEARNING" paper. This paper aims at showcasing the 
interest of using a machine learning approach to allow better prediction of
patients at risk of post infarct left ventricular remodeling than traditional
statistical approaches.

"""


"""
###############################################################################
# Imports and data loading
###############################################################################
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.experimental import enable_iterative_imputer # needs it even if warning that unused
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import joblib
from sklearn.preprocessing import  RobustScaler #, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from skrebate import MultiSURF #, MultiSURFstar
from boruta import BorutaPy
from sklearn.linear_model import ElasticNetCV
import shap

# import custom made packages for ease of use ML functions
from ezml.dataset_manager_v3 import DatasetManagerV3
from ezml.preprocessing import plot_df_nan, nan_filter
from ezml.model_maker import make_model
from ezml.model_training import EzPipeline
from ezml.misc import plot_confusion_matrix


# TODO
# 1. set here the root to the datasets folders
DATASET_PATH = r"D:\Anaconda datasets\BigData\cardiologia\data"
output_path = r"D:\Anaconda datasets\BigData\cardiologia\output"


DATASET_NAME = "cardio_201211_regular" # CLASSIFICATION, WITH BIOLOGY DATA
    
# load the dataset
dataset = DatasetManagerV3(os.path.join(DATASET_PATH, DATASET_NAME))

# get the training/test sets
X_train, y_train = dataset.getTrainingSet()
X_test, y_test = dataset.getTestSet()
dataset._raw_data

# transforming them into dataframes/Series
X_train = pd.DataFrame(X_train, columns=dataset.getColnames())
X_test = pd.DataFrame(X_test, columns=dataset.getColnames())
y_train = pd.DataFrame(y_train, columns=dataset.getColnames("y"))['LV_DILATION_CLASS']
y_test = pd.DataFrame(y_test, columns=dataset.getColnames("y"))['LV_DILATION_CLASS']

# removing unnecessary features
# OG_volume has more missing values than surface
# and masse_DE is the same parameter as infarct_size
# removing M0_NR since M0_NR v2 is better for no reflow assessment
# also removing successful angioplasty since it is actually known with TIMI score

X_train.drop(['M0_OG_VOLINDEX', 'M0_MASSE_TOTAL_DE_INDEX', 'M0_MASSENR', 'CORO_ATC_SAUVETAGE'], axis=1, inplace=True)
X_test.drop(['M0_OG_VOLINDEX', 'M0_MASSE_TOTAL_DE_INDEX', 'M0_MASSENR', 'CORO_ATC_SAUVETAGE'], axis=1, inplace=True)


"""
###############################################################################
# First, Imputing missing values
###############################################################################
"""
# let's check the amount of missing values on our features

_, dropped0 = nan_filter(X_train, threshold=-0.01)

tmp = pd.DataFrame.from_dict(dropped0, orient='index')
tmp = tmp*100
tmp.to_excel(os.path.join(DATASET_PATH, DATASET_NAME, 'missing_values_v3.xlsx'))


# a visual plot first 
plot_df_nan(X_train)
# looks like some features have a very high amount of missing values
_, dropped20 = nan_filter(X_train, threshold=0.20) 

X_train.drop(dropped20.keys(), axis=1, inplace=True)
X_test.drop(dropped20.keys(), axis=1, inplace=True)

colnames = list(X_train.columns)

_, dropped0 = nan_filter(X_train, threshold=-0.01)

np.median(list(dropped0.values()))
np.mean(list(dropped0.values()))


# now let's impute the missing values with an iterative imputer

# before that we need to store what features are categorical (i.e. one hot encoded)
cat_col = list()
for col in X_train.columns :
    if len(X_train.loc[:, col].value_counts()) == 2 :
        cat_col.append(col)
for col in cat_col :
    print(X_train.loc[:,col].value_counts()) # checking if no mistake == OK

pd.Series(cat_col).to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'cat_col.csv'), index=False)

# now let's impute with an ExtraTreesRegressor in an Iterative imputer
ET_imputer = ExtraTreesRegressor(random_state=42)
IT_imputer = IterativeImputer(ET_imputer, random_state=42)
X_train = IT_imputer.fit_transform(X_train)
X_test = IT_imputer.transform(X_test)

# saving the imputer in case of later uses
joblib.dump(IT_imputer, os.path.join(output_path, DATASET_NAME+'_IT_imputer.joblib'))

# retransforming in dataframe
X_train = pd.DataFrame(X_train, columns=colnames)
X_test = pd.DataFrame(X_test, columns=colnames)

# next, we need to correct the categorical imputed values to force them into one hot again
for col in cat_col :
    for i in X_train.index :
        if X_train.loc[i, col] < 0.5 :
            X_train.loc[i, col] = float(0)
        else :
            X_train.loc[i, col] = float(1)            
    for i in X_test.index :
        if X_test.loc[i, col] < 0.5 :
            X_test.loc[i, col] = float(0)
        else :
            X_test.loc[i, col] = float(1)            

# final check of the absence of missing values :
X_train.isna().sum().sum() # 0
X_test.isna().sum().sum() # 0

# exports of imputed dataset
X_train.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed.csv'), index=False)
X_test.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed.csv'), index=False)

# now the datasets are complete


"""
###############################################################################
# Secondly, Feature Scaling
###############################################################################
"""

# let's load the imputed dataset
X_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed.csv'))
X_test = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed.csv'))

# now let's proceed to feature scaling 
# we will compare for each feature the histogram of before and after scaling 

cat_col = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'cat_col.csv'))
cat_col = list(cat_col['0'])
num_col = [col for col in X_train.columns if col not in cat_col]

X_train_base = X_train.copy()
scaler = RobustScaler(quantile_range=(5.0,95.0))

full_pipeline = ColumnTransformer(transformers= [
        ('num_scaler', scaler, num_col)
        ], remainder='passthrough')


X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)

X_train = pd.DataFrame(X_train, columns=colnames)
X_test = pd.DataFrame(X_test, columns=colnames)
joblib.dump(full_pipeline, os.path.join(output_path, DATASET_NAME+'_Scaler.joblib'))

# now onto visualization, let's plot some features distribution to check if ok
for i in range(0, 10) :
    plt.figure()
    X_train_base.loc[:,X_train_base.columns[i]].hist()
    plt.title(X_train_base.columns[i])
    plt.figure()
    plt.title(X_train.columns[i])
    X_train.loc[:,X_train.columns[i]].hist()
    
# Scaling seems satisfying with Robust Scaler and 5-95 quantiles on num_col only

# let's export the scaled data :
X_train.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled.csv'), index=False)
X_test.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled.csv'), index=False)



"""
###############################################################################
# Thirdly, a first broad Feature Selection
###############################################################################
"""

# loading
X_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled.csv'))
X_test = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled.csv'))
colnames = list(X_train.columns)

# iterating with our 3 methods until we have a stable subset of selected features
stable = False
while not stable :
    num_feat_pre = len(colnames)
    # Lasso (or elastic net depending)
    clf = ElasticNetCV(l1_ratio=[0.5, 0.8, 0.9, 0.99, 1], selection='cyclic',max_iter=1000, cv=10, n_jobs=-1) 
    clf.fit(X_train, y_train)
    EN_features = pd.Series(data=clf.coef_, index=colnames).sort_values(ascending = False)
    
    # BorutaPy
    clf = ExtraTreesClassifier(n_jobs=-1, max_depth=5, random_state=42) # 
    selector = BorutaPy(clf, n_estimators=500, perc=100, verbose=2, max_iter=30, random_state=42) 
    selector.fit(X_train.values, y_train)
    # features list
    boFeatures = pd.Series(colnames)[selector.support_]
    boFeatures_rank = pd.Series(colnames)[selector.ranking_]
    boFeatures_weak = pd.Series(colnames)[selector.support_weak_]
    print('{} features selected by BorutaPy'.format(len(boFeatures)),'BorutaPy-selected features are: ', boFeatures)
    
    # RELIEF based methods 
    MS = MultiSURF()
    MS.fit(X_train.values, y_train)
    MS_features = pd.Series(data=MS.feature_importances_, index=colnames).sort_values(ascending = False)
    
    
    ### We will keep only selected features across the different selectors
    EN_features = EN_features[EN_features != 0]
    MS_features = MS_features[MS_features > 0]
    feat_list = pd.Series(list(EN_features.index)+list(MS_features.index)+list(boFeatures))
    feat_to_keep = feat_list.drop_duplicates()
    
    # keeping only the preselected features
    X_train = X_train.loc[:, feat_to_keep]
    X_test = X_test.loc[:, feat_to_keep]
    colnames = list(X_train.columns)

    num_feat_post = len(colnames)
    
    if num_feat_pre == num_feat_post :
        stable = True

# exporting the full subset    
X_train.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled_large_selection.csv'), index=False)
X_test.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled_large_selection.csv'), index=False)

EN_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'EN_features_large.csv'))
MS_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'MS_features_large.csv'))
boFeatures.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'bo_features_large.csv'))
 
EN_features_tmp = abs(round(EN_features, 2)).sort_values(ascending=False)
EN_features_tmp.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'EN_features_large.csv'))
MS_features = round(MS_features, 2)
MS_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'MS_features_large.csv'))


# lets' refit ENET and MS on the 34 informative features to get all feat importances
X_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled_large_selection.csv'))
colnames = list(X_train.columns)
# Lasso (or elastic net depending)
clf = ElasticNetCV(l1_ratio=[0.5, 0.8, 0.9, 0.99, 1], selection='cyclic',max_iter=1000, cv=10, n_jobs=-1) 
clf.fit(X_train, y_train)
EN_features = pd.Series(data=clf.coef_, index=colnames).sort_values(ascending = False)

# RELIEF based methods 
MS = MultiSURF()
MS.fit(X_train.values, y_train)
MS_features = pd.Series(data=MS.feature_importances_, index=colnames).sort_values(ascending = False)

EN_features_tmp = abs(round(EN_features, 3)).sort_values(ascending=False)
EN_features_tmp.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'EN_features_full.csv'))
MS_features = round(MS_features, 3)
MS_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'MS_features_large_full.csv'))


# Selecting only top5 for more stringency (given the amount of power we have with 
# the size of our cohort)
EN_features = abs(EN_features).sort_values(ascending=False)[:5]
MS_features = MS_features[:5]
feat_list = pd.Series(list(EN_features.index)+list(MS_features.index)) #+list(boFeatures))
feat_to_keep = feat_list.drop_duplicates()

# down to 7 features features

X_train_5 = X_train.loc[:, feat_to_keep]
X_test_5 = X_test.loc[:, feat_to_keep]

X_train_5.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled_select.csv'), index=False)
X_test_5.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled_select.csv'), index=False)

EN_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'EN_features.csv'))
MS_features.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'MS_features.csv'))



"""
###############################################################################
# Exploring the relevant features (eventual feature engineering and final choice)
###############################################################################
"""
X_train_5 = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled_select.csv'))
X_test_5 = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled_select.csv'))


# Visualization with scatter mattrix and corr matrix
pd.plotting.scatter_matrix(X_train_5)

corr_matrix = X_train_5.corr()
# no excessively correlated features here so we can keep all features without 
# fear of colinearity


"""
###############################################################################
# Train/val split before training and comparing models
###############################################################################
"""
# we will start from  the X_train_5 features with 7 features
X_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled_select.csv'))

X_train_full = X_train.copy()
y_train_full = y_train.copy()

# split between train and validation data
from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, 
                               test_size=0.2, 
                               random_state=2)

for train_index, val_index in split.split(X_train_full, y_train_full):
    X_val = X_train_full.loc[val_index]
    X_train = X_train_full.loc[train_index]
    y_val = y_train_full.loc[val_index]
    y_train = y_train_full.loc[train_index]

X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)


X_train.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_final7.csv'), index=False)
X_val.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_val_final7.csv'), index=False)
y_train.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'y_train_final7.csv'), index=False)
y_val.to_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'y_val_final7.csv'), index=False)



"""
###############################################################################
# training and validating Best Models 
###############################################################################
"""

# loading the final processed data
X_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_final7.csv'))
X_val = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_val_final7.csv'))
X_test = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled_select.csv'))
y_train = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'y_train_final7.csv'))
y_val = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'y_val_final7.csv'))
colnames = list(X_train.columns)

y_train = pd.Series(y_train.values.ravel())
y_val = pd.Series(y_val.values.ravel())

# Preprocessing already done

# Trying a SGD logistic classifier
sgd_model = make_model(models=['SGD'],
                       class_weight=[None, 'balanced'],
                       C = [0.01, 0.1, 0.5, 1, 2, 5, 10],
                       loss = ['log', 'modified_huber'],
                       penalty = ['l2', 'elasticnet'],
                       alpha = [0.0001, 0.001, 0.01, 0.00001],
                       l1_ratio = [0, 0.5, 0.8, 0.9, 1],
                       max_iter = [1000],
                       tol = [0.001],
                       n_jobs=-1,
                       random=0)

sgd = EzPipeline(X_train, y_train, X_val, y_val)

sgd_clf = sgd.train_classifier(
                      sgd_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

sgd.evaluate_classifier(sgd_clf, 
                    sgd.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = True,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = sgd_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,1]))
y_val_ = sgd_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,1]))


# Trying a linearSVM classifier
linSVC_model = make_model(models=['LinearSVM'],
                       class_weight=[None, 'balanced'],
                       C = [0.01, 0.1, 0.5, 1, 2, 5, 10],
                       n_jobs=-1,
                       random=0)

linSVC = EzPipeline(X_train, y_train, X_val, y_val)

linSVC_clf = linSVC.train_classifier(
                      linSVC_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

linSVC.evaluate_classifier(linSVC_clf, 
                    linSVC.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = True,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = linSVC_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,0]))
y_val_ = linSVC_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,0]))


# Trying a rbfSVM classifier
SVC_model = make_model(models=['SVM'],
                         class_weight=[None, 'balanced'],
                         C = [0.01, 0.1, 0.5, 1, 2, 5, 10],
                         n_jobs=-1,
                         random=0)

SVC = EzPipeline(X_train, y_train, X_val, y_val)

SVC_clf = SVC.train_classifier(
                      SVC_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

SVC.evaluate_classifier(SVC_clf, 
                    SVC.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = True,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = SVC_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,0]))
y_val_ = SVC_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,0]))


# Extra_trees classifier
ET_model = make_model(models=['ExtraTrees'],
                         class_weight=[None, 'balanced'],
                         max_depth=[1, 3, 5, None],
                         min_samples_split=[2, 5, 10],
                         min_samples_leaf=[1, 2, 3],
                         n_estimators=[20,500, 1000],
                         n_jobs=-1,
                         random=0)

ET = EzPipeline(X_train, y_train, X_val, y_val)

ET_clf = ET.train_classifier(
                      ET_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

ET.evaluate_classifier(ET_clf, 
                    ET.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = True,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = ET_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,1]))
y_val_ = ET_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,1]))


# LightGBM classifier
LGBM_model = make_model(models=['LightGBM'],
                         num_leaves = [5, 10, 31],
                         n_estimators_gb=[3, 4, 5, 10, 50, 100],
                         learning_rate_gb = [0.01, 0.1],
                         gamma_gb = [0, 1, 5, 10],
                         max_depth_gb = [1, 5, -1],
                         min_child_weight = [0.001],
                         lambda_gb = [0, 0.01],
                         alpha_gb = [0, 0.01],
                         subsample = [1],
                         colsample_bytree = [1],
                         n_jobs=-1,
                         random=0)

LGBM = EzPipeline(X_train, y_train, X_val, y_val)

LGBM_clf = LGBM.train_classifier(
                      LGBM_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

LGBM.evaluate_classifier(LGBM_clf, 
                    LGBM.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = True,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = LGBM_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,1]))
y_val_ = LGBM_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,1]))



# DNN model
### Let's also try a wide and deep NN architecture
# It will require us to use Keras functional API 
# as well as to code everything directly 

# specific imports
from tensorflow.keras.layers import Input, Dense, Concatenate, AlphaDropout
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC as keras_AUC
from tensorflow.keras.optimizers import Nadam  
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf


# making sure no models are stored in session()
K.clear_session()
seed_ = 0
init = lecun_normal(seed=seed_)
class_weight = True

np.random.seed(2) 
tf.random.set_seed(2)



# model architecture
input_ = Input(shape=[7], name='input')
hidden_1 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(input_)
dropout_layer_1 =  AlphaDropout(rate=0.1, seed=seed_)(hidden_1)
concat = Concatenate()([input_, dropout_layer_1])
output = Dense(1, activation='sigmoid', kernel_initializer='lecun_normal', name='main_output')(concat)

model = Model(inputs=input_,
              outputs=output)

model.compile(loss='binary_crossentropy',
              #loss_weights={'main_output':0.9,'deep_output':0.1},
              optimizer=Nadam(learning_rate=0.01),
              metrics=[keras_AUC()])

# callbacks
outdir = output_path
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
savedir = os.path.join(outdir,'run-{}'.format(now),'model.h5')
logdir = os.path.join(outdir,'run-{}'.format(now))

callbacks = [ModelCheckpoint(filepath=savedir, monitor='val_loss', save_best_only=True, mode='min'),
             TensorBoard(log_dir=logdir),
             EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=1, mode='min', restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.001, min_lr=0)]


# Weighting classes (for imbalanced data if submitted)
if class_weight :
    class_weight = dict()
    for num_class in range(len(pd.value_counts(y_train))) :
        class_weight[int(num_class)] = (1 / (np.unique(y_train, return_counts=True)[1][num_class]))*(len(y_train))/2.0
else :
    class_weight = None

# fitting model
history = model.fit(X_train,
                    y_train, 
                    batch_size=32, 
                    epochs=1000, 
                    validation_data=(X_val, y_val),
                    class_weight = class_weight,
                    callbacks = callbacks
                    )


# best model loading (since it is our best model so far we can test it on test set)
model = load_model(r"D:\Anaconda datasets\BigData\cardiologia\output\run-20210708164528\model.h5")


model.summary()

# we could use MonteCarlo dropout for evaluation (as well on test set since it is the best model)
y_probas_train = np.stack([model(X_train.values, training=True) for sample in range(1000)])
y_proba_train = y_probas_train.mean(axis=0)
y_probas_val = np.stack([model(X_val.values, training=True) for sample in range(1000)])
y_proba_val = y_probas_val.mean(axis=0)
y_probas_test = np.stack([model(X_test.values, training=True) for sample in range(1000)])
y_proba_test = y_probas_test.mean(axis=0)

# or use standard predict (let's choose this one)
y_proba_train = model.predict(X_train.values)
y_proba_val = model.predict(X_val.values)
y_proba_test = model.predict(X_test.values)

print('train auc is :', roc_auc_score(y_train, y_proba_train)) 
print('val auc is :', roc_auc_score(y_val, y_proba_val)) 
print('test auc is :', roc_auc_score(y_test, y_proba_test)) 

# to check on val
#y_proba_val = model.predict(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_proba_val)

# to check on test
fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)


"""
###############################################################################
# TESTING BASELINE MODEL AND COMPARING WITH BEST MODEL
###############################################################################
"""

# loading
X_train_sgd = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_train_imputed_scaled.csv'))
X_test_sgd = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME, 'X_test_imputed_scaled.csv'))
colnames = list(X_train.columns)


# we need to test a reglog with "classic" cardiological variable as a baseline model
# with which to compare our best model

cols = ['MAX_CK', 'DELAI_DOULEURATC_MINUTES', 'M0_FEVG', 'INFARCT_SIZE_',
        'HTA__0NON_1OUI', 'DIABETE__0NON_1OUI', 'CHOLESTEROL__0NON_1OUI',
        'HEREDITE__1OUI_0NON', 'HOSP_IVG__1OUI_0NON', 'TABAC__NON_FUMEUR_OU_SEVRE0_FU',
        'M0_NRv2']

X_train_sgd = X_train_sgd.loc[:,cols]
X_test_sgd = X_test_sgd.loc[:,cols]

# splitting into train and val data to stay identical to our previous pipeline
X_train_full_sgd = X_train_sgd.copy()
y_train_full_sgd = y_train.copy()

from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, 
                               test_size=0.2, 
                               random_state=2)

for train_index, val_index in split.split(X_train_full_sgd, y_train_full_sgd):
    X_val_sgd = X_train_full_sgd.loc[val_index]
    X_train_sgd = X_train_full_sgd.loc[train_index]
    y_val_sgd = y_train_full_sgd.loc[val_index]
    y_train_sgd = y_train_full_sgd.loc[train_index]

X_train_sgd.reset_index(drop=True, inplace=True)
X_val_sgd.reset_index(drop=True, inplace=True)
y_train_sgd.reset_index(drop=True, inplace=True)
y_val_sgd.reset_index(drop=True, inplace=True)


# Trying a SGD logistic classifier
sgd_model = make_model(models=['SGD'],
                       class_weight=[None, 'balanced'],
                       C = [0.01, 0.1, 0.5, 1, 2, 5, 10],
                       loss = ['log', 'modified_huber'],
                       penalty = ['l2', 'elasticnet'],
                       alpha = [0.0001, 0.001, 0.01, 0.00001],
                       l1_ratio = [0, 0.5, 0.8, 0.9, 1],
                       max_iter = [1000],
                       tol = [0.001],
                       n_jobs=-1,
                       random=0)

sgd = EzPipeline(X_train, y_train, X_val, y_val)

sgd_clf = sgd.train_classifier(
                      sgd_model,
                      list(X_train.columns),
                      outdir = output_path,
                      random_seed = 0,
                      under_sampling = None,
                      rus_strategy = 0.5,
                      grid_search = True,
                      grid_search_splits = 10,
                      grid_search_test_size = 0.2,
                      rfe = False,
                      rfe_step = 1,
                      min_rfe = 5,
                      n_jobs = -2)

sgd.evaluate_classifier(sgd_clf, 
                    sgd.selected_feats_, 
                    outdir = output_path, 
                    title = 'M3 Left Ventricular Remodelling prediction (val set)', 
                    dpi = 600, 
                    extra_metrics = False,
                    use_shap  = False,                           
                    shap_type = 'kernel',
                    dependence_plot = False)

y_train_ = sgd_clf.predict_proba(X_train)                      
print('train auc is :', roc_auc_score(y_train, y_train_[:,1]))
y_val_ = sgd_clf.predict_proba(X_val)                      
print('val auc is :', roc_auc_score(y_val, y_val_[:,1]))
y_test_ = sgd_clf.predict_proba(X_test)                      
print('test auc is :', roc_auc_score(y_test, y_test_[:,1]))


#train auc is : 0.7103923330338424
#val auc is : 0.6522435897435899
#test auc is : 0.708984375

"""
###############################################################################
# PLOTTING CONFUSION MATRIX ON BEST THRESHOLD, allowing calculation of 
# Sensitivity and Specificity
###############################################################################
"""

# getting the y_pred and y_test right
for i in range(len(y_test)) :
    if y_test.loc[i] == 0 : 
        y_test.loc[i] = 'No LVR'
    else :
        y_test.loc[i] = 'LVR'

# to plot for best model reglog
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred.ravel())
threshold = 0.32654
for i in range(len(y_pred)) :
    if y_pred.loc[i] < threshold : 
        y_pred.loc[i] = 'No LVR'
    else :
        y_pred.loc[i] = 'LVR'
    
plot_confusion_matrix(y_test, y_pred, labels=['No LVR','LVR'])


# to plot for baseline reglog
y_pred = sgd_clf.predict_proba(X_test)
y_pred = pd.Series(y_pred[:,1].ravel())
threshold = 0.25085
for i in range(len(y_pred)) :
    if y_pred.loc[i] < threshold : 
        y_pred.loc[i] = 'No LVR'
    else :
        y_pred.loc[i] = 'LVR'
    
plot_confusion_matrix(y_test, y_pred, labels=['No LVR','LVR'])


"""
###############################################################################
# PLOTTING ROC curves
###############################################################################
"""


# to import baseline sgd_reglog
sgd_clf_path = r"D:\Anaconda datasets\BigData\cardiologia\output\train_classifier_SGDClassifier_20210601083938.joblib"

from joblib import load
sgd_clf = load(sgd_clf_path)


model = load_model(r"D:\Anaconda datasets\BigData\cardiologia\output\run-20210708164528\model.h5")

# to plot ROC curves with both baseline and best model on val and then test sets
y_proba_val_dnn = model.predict(X_val.values)
fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(y_val, y_proba_val_dnn)

y_proba_val_sgd = sgd_clf.predict_proba(X_val_sgd)
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_val_sgd, y_proba_val_sgd[:,1])


# plot (Figure 1)
title = 'M3 Left Ventricular Remodelling prediction (val set)'
dpi = 1000

roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
youden_dnn = (1-fpr_dnn)+tpr_dnn-1
best_threshold_dnn = np.where(youden_dnn == np.max(youden_dnn))[0][0]
severe_threshold_dnn = np.argmin(np.abs(thresholds_dnn-.9))

roc_auc_sgd = auc(fpr_sgd, tpr_sgd)
youden_sgd = (1-fpr_sgd)+tpr_sgd-1
best_threshold_sgd = np.where(youden_sgd == np.max(youden_sgd))[0][0]
severe_threshold_sgd = np.argmin(np.abs(thresholds_sgd-.9))


plt.figure(figsize=(8,8))

plt.plot(fpr_dnn, tpr_dnn, color='purple',
         lw=2, label='ROC curve ML (area = %0.2f)' % roc_auc_dnn)
plt.plot(fpr_dnn[best_threshold_dnn], tpr_dnn[best_threshold_dnn], 'o', color = 'red')

plt.plot(fpr_sgd, tpr_sgd, color='darkorange',
         lw=2, label='ROC curve baseline (area = %0.2f)' % roc_auc_sgd)
plt.plot(fpr_sgd[best_threshold_sgd], tpr_sgd[best_threshold_sgd], 'o', color = 'red')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC - '+title)
plt.legend(loc="lower right")
tmp_sub=0.01

plt.text(fpr_dnn[best_threshold_dnn]+.01, tpr_dnn[best_threshold_dnn]+0, 'Optimal: ' + str(round(thresholds_dnn[best_threshold_dnn], 5)) +
         ' (Se: ' + str(str(round(100*tpr_dnn[best_threshold_dnn], 1))) + ', Sp: ' + str(str(round(100*(1-fpr_dnn[best_threshold_dnn]), 1))) + ')')
plt.text(fpr_sgd[best_threshold_sgd]+.01, tpr_sgd[best_threshold_sgd]+0, 'Optimal: ' + str(round(thresholds_sgd[best_threshold_sgd], 5)) +
         ' (Se: ' + str(str(round(100*tpr_sgd[best_threshold_sgd], 1))) + ', Sp: ' + str(str(round(100*(1-fpr_sgd[best_threshold_sgd]), 1))) + ')')


from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
outpath = os.path.join(r'D:\Anaconda datasets\BigData\cardiologia\REMOVE\IJC','ROC_plot_val_{}.png'.format(now))

plt.savefig(outpath, dpi=dpi)


# to plot on test
y_proba_test_dnn = model.predict(X_test.values)
fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(y_test, y_proba_test_dnn)

y_proba_test_sgd = sgd_clf.predict_proba(X_test_sgd)
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_test, y_proba_test_sgd[:,1])


# plot (Figure 1)
title = 'M3 Left Ventricular Remodelling prediction (test set)'
dpi = 1000

roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
youden_dnn = (1-fpr_dnn)+tpr_dnn-1
best_threshold_dnn = np.where(youden_dnn == np.max(youden_dnn))[0][0]
severe_threshold_dnn = np.argmin(np.abs(thresholds_dnn-.9))

roc_auc_sgd = auc(fpr_sgd, tpr_sgd)
youden_sgd = (1-fpr_sgd)+tpr_sgd-1
best_threshold_sgd = np.where(youden_sgd == np.max(youden_sgd))[0][0]
severe_threshold_sgd = np.argmin(np.abs(thresholds_sgd-.9))


plt.figure(figsize=(8,8))

plt.plot(fpr_dnn, tpr_dnn, color='purple',
         lw=2, label='ROC curve ML (area = %0.2f)' % roc_auc_dnn)
plt.plot(fpr_dnn[best_threshold_dnn], tpr_dnn[best_threshold_dnn], 'o', color = 'red')

plt.plot(fpr_sgd, tpr_sgd, color='darkorange',
         lw=2, label='ROC curve baseline (area = %0.2f)' % roc_auc_sgd)
plt.plot(fpr_sgd[best_threshold_sgd], tpr_sgd[best_threshold_sgd], 'o', color = 'red')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC - '+title)
plt.legend(loc="lower right")
tmp_sub=0.01

plt.text(fpr_dnn[best_threshold_dnn]+.01, tpr_dnn[best_threshold_dnn]+0, 'Optimal: ' + str(round(thresholds_dnn[best_threshold_dnn], 5)) +
         ' (Se: ' + str(str(round(100*tpr_dnn[best_threshold_dnn], 1))) + ', Sp: ' + str(str(round(100*(1-fpr_dnn[best_threshold_dnn]), 1))) + ')')
plt.text(fpr_sgd[best_threshold_sgd]+.01, tpr_sgd[best_threshold_sgd]+0, 'Optimal: ' + str(round(thresholds_sgd[best_threshold_sgd], 5)) +
         ' (Se: ' + str(str(round(100*tpr_sgd[best_threshold_sgd], 1))) + ', Sp: ' + str(str(round(100*(1-fpr_sgd[best_threshold_sgd]), 1))) + ')')


from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
outpath = os.path.join(r'D:\Anaconda datasets\BigData\cardiologia\REMOVE\IJC','ROC_plot_val_{}.png'.format(now))

plt.savefig(outpath, dpi=dpi)


"""
###############################################################################
# PLOTTING SHAP plot
###############################################################################
"""

# now also outputting shap plots

# renaming features 
X_train.rename(columns={'MED_CK':'Creatine Kinase (med)',
                        'DIABETE__0NON_1OUI':'Diabetes',
                        'M0_OG_SURF':'Left Atrial Surface',
                        'MAX_IDR':'RBC distribution width (max)',
                        'MAX_VGM':'Mean corpuscular volume (max)',
                        'MAX_CREAT':'Creatininemia (max)',
                        'HTA__0NON_1OUI':'Hypertension',}, inplace=True)

# creating a function for shap to call the MC dropout model prediction
def MCdropout_pred(x) :
    return np.stack([model(x, training=True) for sample in range(1000)]).mean(axis=0)

X_train_summary = shap.kmeans(X_train, 30)                    
#shap_values = shap.KernelExplainer(MCdropout_pred, X_train_summary).shap_values(X_train)
shap_values = shap.KernelExplainer(model, X_train_summary).shap_values(X_train)
shap_values = shap.KernelExplainer(model, X_train).shap_values(X_train)
# making summary_plot
summaryplot = plt.figure()
shap.summary_plot(shap_values[0], X_train)

summaryplot.savefig(os.path.join(outdir,'evaluate_classifier_summary_plot_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)

# making dependence_plot
for feat in list(X_train.columns) :
    dependenceplot = plt.figure()
    shap.dependence_plot(list(X_train.columns).index(feat), shap_values[0], X_train, feature_names=list(X_train.columns))
    
    dependenceplot.savefig(os.path.join(outdir, 'evaluate_classifier_dependance_plot_'+str(feat)+'_{}.png'.format(now)), bbox_inches='tight', dpi=dpi)
        
print('evaluation plots have been written in ', outpath)


## END of analysis


