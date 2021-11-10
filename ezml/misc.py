# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:30:42 2020

@author: Xavier Dieu

===============================================================================
Miscellaneous FUNCTIONS AND CLASSES FOR EZML
===============================================================================
# some may still not work !

"""

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# FUNCTIONS

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


def save_fig(fig_id, IMAGES_PATH, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)




# CLASSES


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

        


