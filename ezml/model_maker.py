# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:30:42 2020

@author: Xavier Dieu

===============================================================================
FUNCTIONS AND CLASSES TO MAKE MODELS FOR EZML
===============================================================================


"""
# IMPORTS
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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
from lightgbm import LGBMClassifier


# FUNCTIONS

def make_model(models, random, custom_models = None,
                     strategy=['stratified', 'most_frequent'],
                     max_depth=[5,10, None],
                     min_samples_split=[2, 5, 8],
                     min_samples_leaf=[1, 2, 3, 5],
                     n_estimators=[20,100, 1000],
                     class_weight=[None, 'balanced'],
                     C = [0.1, 0.5, 1, 2, 5],
                     n_neighbors = [3,5,10],
                     weights = ['uniform', 'distance'],
                     num_leaves = [10, 31],
                     n_estimators_gb=[3, 4, 5, 10, 50, 100],
                     learning_rate_gb = [0.01, 0.1, 0.3],
                     gamma_gb = [0, 1, 5, 10],
                     max_depth_gb = [1, 2, 3, 4, 10, 20],
                     min_child_weight = [0.5, 1, 2],
                     lambda_gb = [1],
                     alpha_gb = [0],
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
                     early_stopping = [True],
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
                      'n_estimators' : n_estimators_gb,
                      'learning_rate' : learning_rate_gb,
                      'gamma' : gamma_gb,
                     'max_depth': max_depth_gb,
                     'min_child_weight': min_child_weight,
                     'lambda' : lambda_gb,
                     'alpha' : alpha_gb,
                     'subsample' : subsample,
                     'colsample_bytree' : colsample_bytree,
                     'colsample_bynode' : colsample_bynode,
                     'scale_pos_weight' : scale_pos_weight
                     }

    if 'LightGBM' in models :
        models_grid[LGBMClassifier(n_jobs=n_jobs, random_state=random)] = {
                        'num_leaves' : num_leaves,
                      'n_estimators' : n_estimators_gb,
                      'learning_rate' : learning_rate_gb,
                     'max_depth': max_depth_gb,
                     'min_child_weight': min_child_weight,
                     'reg_lambda' : lambda_gb,
                     'reg_alpha' : alpha_gb,
                     'subsample' : subsample,
                     'colsample_bytree' : colsample_bytree,
                     'class_weight' : class_weight
                     }

    if 'LinearSVM' in models :
        models_grid[SVC(kernel='linear', probability=True, random_state=random)] = {
                     'C': C,
                     'class_weight': class_weight
                     }

    if 'SGD' in models :
        models_grid[SGDClassifier(random_state=random, n_jobs=n_jobs)] = {
                     'loss': loss,
                     'penalty': penalty,
                     'alpha': alpha,
                     'l1_ratio': l1_ratio,
                     'max_iter': max_iter,
                     'tol': tol,
                     'class_weight': class_weight,
                     'early_stopping': early_stopping
                     }

    if custom_models != None :
        for key in custom_models.keys() :
            models_grid[key] = custom_models[key]

    
    return models_grid



def make_DNNmodel(input_dim,
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





# CLASSES


