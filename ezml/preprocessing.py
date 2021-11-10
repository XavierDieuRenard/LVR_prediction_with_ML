# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:30:42 2020

@author: Xavier Dieu

===============================================================================
PREPROCESSING FUNCTIONS FOR EZML
===============================================================================


"""
# IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import tensorflow_addons as tfa

# FUNCTIONS

def stratified_splitter(df, test_size, random_state, target,
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
        print('stratified sampling results in train set : ', '\n', 
              df[target].value_counts()/len(df),
              '\n',
              '\n'
              'stratified sampling results in test set : ', '\n',
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



def prep_data_saver(INPUT_PATH, data_set, path_col, name):
    """
    reorganizing data into desired directory"
    """
    # Where to save/move the data
    PATH = os.path.join(INPUT_PATH, name)
    os.makedirs(PATH, exist_ok=True)
    
    # 
    for i in data_set.index :
        filename = data_set.loc[i, path_col].split(os.sep)[-1] 
        os.rename(data_set.loc[i,path_col], os.path.join(PATH, filename))
        print('...Saving {} {} data : {}/{}'.format(name, path_col, i+1, len(data_set)))
        
    return None



def update_path(PATH, data_set, cols) :
    
    for i in data_set.index :
        for col in cols :
            filename = os.path.basename(data_set.loc[i, col])
            data_set.loc[i, col] = os.path.join(PATH, filename)
    return data_set



def get_img_label(filepath, data_set):
    # map the path to its target
    ds = data_set.filter(lambda x, y  : x == filepath)
    label = 0.
    for i, j in ds : 
        label = j
    return label   



def get_img_mask(filepath, data_set, mask_ext):
    # map the path to its target
    ds = data_set.filter(lambda x, y  : x == filepath)
    mask = 0.
    for i, j in ds : 
        mask = load_image(j, mask_ext)
    return mask   



def load_image(filepath, ext):
    image = tf.io.read_file(filepath)
    if ext == '.bmp':
        image = tf.image.decode_bmp(image)
    elif ext == '.png':
        image = tf.image.decode_png(image)        
    elif ext == '.jpeg':
        image = tf.image.decode_jpeg(image)        
    else :
        raise ValueError
        print('incorrect file ext')
    image = tf.image.convert_image_dtype(image, tf.float32)        
    return image



def load_img_from_path(filepath, data_set, ext):
    label = get_img_label(filepath, data_set)
    image = load_image(filepath, ext)
    return image, label



def load_mask_from_path(filepath, data_set, ext, mask_ext):
    mask = get_img_mask(filepath, data_set, mask_ext)
    image = load_image(filepath, ext)
    return image, mask



def resize_rescale(image, height, width, scale) :
    
    
    resize = tf.keras.layers.experimental.preprocessing.Resizing(height, width)
    image = resize(image)
    
    if tf.reduce_max(image) <= 1. and tf.reduce_min(image) >= -1. :
        print('image already scaled')
    elif scale == '01' :
        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) 
        image = rescale(image)
    elif scale == '-11' :
        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
        image = rescale(image)
        
    else : 
        print('no scaling applied')
    
    print("Min and max pixel values:", tf.reduce_min(image), tf.reduce_max(image))
    
    return image



def resize_rescale_withmask(image, mask, height, width, scale) :
    
    
    resize = tf.keras.layers.experimental.preprocessing.Resizing(height, width)
    image = resize(image)
    mask = resize(mask)
    
    if tf.reduce_max(image) <= 1. and tf.reduce_min(image) >= -1. :
        print('image already scaled')
    elif scale == '01' :
        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) 
        image = rescale(image)
        mask = rescale(mask)
    elif scale == '-11' :
        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
        image = rescale(image)
        mask = rescale(mask)        
    else : 
        print('no scaling applied')
    
    print("Min and max pixel values:", tf.reduce_min(image), tf.reduce_max(image))
    
    return image, mask



def data_augment_spatial(image, mode = 'both'):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if mode == 'both' :
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    if mode == 'hor' :
        image = tf.image.random_flip_left_right(image)
    if mode == 'ver' :
        image = tf.image.random_flip_left_right(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    return image



def data_augment_rotate(image, max_deg):
    rot = tf.random.uniform([], 0, max_deg, dtype=tf.float32)
    
    image = tfa.image.rotate(image, rot)

    return image



def data_augment_shear(image, level):
    p_shear = tf.random.uniform([], 0, 1, dtype=tf.float32)
    
    if p_shear > 0.5 : 
        image = tfa.image.shear_x(image, level, replace=tf.constant(0))
        image = tfa.image.shear_y(image, level, replace=tf.constant(0))
    elif p_shear > 0.25 :
        image = tfa.image.shear_y(image, level, replace=tf.constant(0))
    else :
        image = tfa.image.shear_x(image, level, replace=tf.constant(0))

    return image



def data_augment_crop(image, img_height, img_width, channels, crop_factor=0.7):
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    crop_size = tf.random.uniform([], int(img_height*crop_factor), img_height, dtype=tf.int32)
    
    if p_crop > .5:
        image = tf.image.random_crop(image, size=[crop_size, crop_size, channels])
    else:
        if p_crop > .4:
            image = tf.image.central_crop(image, central_fraction=.7)
        elif p_crop > .2:
            image = tf.image.central_crop(image, central_fraction=.8)
        else:
            image = tf.image.central_crop(image, central_fraction=.9)
    
    image = tf.image.resize(image, size=[img_height, img_width])

    return image



def data_augment_cutout(image, mask_size, n_cut_max=50): 
    
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if p_cutout > .75: 
        n_cutout = tf.random.uniform([], int(n_cut_max/2), int(n_cut_max), dtype=tf.int32)
        for k in range(n_cutout) :
            image = tfa.image.random_cutout(image, mask_size)
    elif p_cutout > .5: 
        n_cutout = tf.random.uniform([], int(n_cut_max/10), int(n_cut_max/2), dtype=tf.int32)
        for k in range(n_cutout) :
            image = tfa.image.random_cutout(image, mask_size)
    elif p_cutout > .25: # 2~5 cut outs
        n_cutout = tf.random.uniform([], int(1), int(n_cut_max/10), dtype=tf.int32)
        for k in range(n_cutout) :
            image = tfa.image.random_cutout(image, mask_size)
    else: # 1 cut out
        image = tfa.image.random_cutout(image, mask_size, 0)

    return image



def data_augment(image, arglist, min_s=0.7, max_s=1.3, min_c=0.8, max_c=1.2, b=0.1):
    """
    Master function that calls many different image augmentations functions

    Parameters
    ----------
    image : tf.tensor
        DESCRIPTION.
    arglist : list
        list of tuples containing args in thef following order
        [(spatial_args),(rotate_args), (crop_args), (cutout_args), (shear_args)]
        probability of occurence must be the first arg stated

    Returns
    -------
    image : tf.tensor 
        image augmented

    """
    # sampling proba of each transformation occurence
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if p_spatial > arglist[0][0]:
        image = data_augment_spatial(image, arglist[0][1])
 
    if p_rotate > arglist[1][0]:
        image = data_augment_rotate(image, arglist[1][1])
    
    if p_crop > arglist[2][0]:
        image = data_augment_crop(image, arglist[2][1], arglist[2][2],
                                  arglist[2][3], arglist[2][4])
            
    image = tf.image.random_saturation(image, min_s, max_s)
    image = tf.image.random_contrast(image, min_c, max_c)
    image = tf.image.random_brightness(image, b)

    image = tf.expand_dims(image, 0)

    if p_cutout > arglist[3][0]:
        image = data_augment_cutout(image, arglist[3][1], arglist[3][2])

    image = tf.squeeze(image, 0)

    if p_shear > arglist[4][0]:
        image = data_augment_shear(image, arglist[4][1])
    
    return image



