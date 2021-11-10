# required libraries
import pandas as pd
import numpy as np
import os
import pathlib
import re
import pickle
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

class DatasetManagerV3:
    """
    
    DatasetManager
    ==============
    
    Description
    -----------
    
    # TODO rewrite
    
    Allows quick and easy management of data frames (pre-procesing, partitionning, missing-data imputation...).
    If path_or_df is a pandas.DataFrame, initializes a new DatasetManager instance based on this dataframe.
    Else, if path_or_df is a string, loads a saved instance of a DatasetManager at the directory pointed by path_or_df.
    
    Parameters
    ----------

    path_or_df : pandas.DataFrame or str
        a dataframe to initialize the dataset manager, or the path to a saved instance of DatasetManager
    
    """
    def __init__(self, path_or_df, seed = 42, check_names = True):
        if isinstance(path_or_df, pd.DataFrame):
            df = path_or_df.copy()
            if check_names:
                colnames = df.columns.tolist()
                
                def strip_accents(s):
                    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
                
                # render column names PERFECT!
                new_colnames = [re.sub("[^a-zA-Z0-9_]+", "", re.sub("[ ]+", "_", strip_accents(str(c)))).upper()[:30] for c in colnames]
                new_colnames = np.array(new_colnames).astype("<U40")
                
                # check duplcicates
                for c in new_colnames:
                    n_replicates = np.sum(new_colnames==c)
                    if n_replicates > 1:
                        new_colnames[np.where(new_colnames==c)[0]]=np.core.defchararray.add(c, np.array(["v{}".format(ci) for ci in list(range(1,n_replicates+1))]))
                        
                # assign
                df.columns = new_colnames
            self._raw_data = df
            self._proc_data = df.copy()
            self._frozen = False
            self._X_train = None
            self._X_test = None
            self._y_train = None
            self._y_test = None
            self._X_mean = None
            self._X_std = None
            self._y_mean = None
            self._y_std = None
            self._seed = seed
            self._train_part = None
            self._test_part = None
            self._columns = []
            self._forbid_multilabel = []
        elif isinstance(path_or_df, str):
            self._raw_data = pd.read_hdf(os.path.join(path_or_df,"dataset.h5"), "raw_data")
            self._proc_data = pd.read_hdf(os.path.join(path_or_df,"dataset.h5"), "proc_data")
            self._frozen = True
            
            self._X_train = np.load(os.path.join(path_or_df,"_X_train.npy"), allow_pickle=True)
            self._X_test = np.load(os.path.join(path_or_df,"_X_test.npy"), allow_pickle=True)
            self._y_train = np.load(os.path.join(path_or_df,"_y_train.npy"), allow_pickle=True)
            self._y_test = np.load(os.path.join(path_or_df,"_y_test.npy"), allow_pickle=True)
            
            self._X_mean = np.load(os.path.join(path_or_df,"_X_mean.npy"), allow_pickle=True)
            self._X_std = np.load(os.path.join(path_or_df,"_X_std.npy"), allow_pickle=True)
            self._y_mean = np.load(os.path.join(path_or_df,"_y_mean.npy"), allow_pickle=True)
            self._y_std = np.load(os.path.join(path_or_df,"_y_std.npy"), allow_pickle=True)
            
            with open(os.path.join(path_or_df,"_seed.pkl"), "rb") as fp:   # Unpickling
                self._seed = pickle.load(fp)
            with open(os.path.join(path_or_df,"_train_part.pkl"), "rb") as fp:   # Unpickling
                self._train_part = pickle.load(fp)
            with open(os.path.join(path_or_df,"_test_part.pkl"), "rb") as fp:   # Unpickling
                self._test_part = pickle.load(fp)
            with open(os.path.join(path_or_df,"_columns.pkl"), "rb") as fp:   # Unpickling
                self._columns = pickle.load(fp)
        else:
            raise Exception("path_or_df should be either a pandas DataFrame or the path towards a saved instance of DatasetManager")

    def save(self, path, name):
        """
        
        Saves an instance of a DatasetManager
        =====================================
        
        :Example:
            
        >>> my_dset.save("C:/Users/admin/downloads","my_dset_v1")
        
        Description
        -----------
            
        This function allows to save the DatasetManager instance, in order to reload it later by initializing a new DatasetManager.
        All parameters will be saved in the path/name directory (created upon save).
        DatasetManager must be frost (function `freezeDataset`) before saving.
        
        Parameters
        ---------
        
        path : str
            the main directory where the subdirectory will be created
            
        name: str
            the subdirectory which will be created inside `path`, inside of which all filles will be stored
        
        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        # create directory
        pathlib.Path(os.path.join(path,name)).mkdir(parents=True, exist_ok=True)
        self._raw_data.to_hdf(os.path.join(path,name,"dataset.h5"), key="raw_data")
        self._proc_data.to_hdf(os.path.join(path,name,"dataset.h5"), key="proc_data")
        np.save(os.path.join(path,name,"_X_train.npy"), self._X_train)
        np.save(os.path.join(path,name,"_X_test.npy"), self._X_test)
        np.save(os.path.join(path,name,"_y_train.npy"), self._y_train)
        np.save(os.path.join(path,name,"_y_test.npy"), self._y_test)
        
        np.save(os.path.join(path,name,"_X_mean.npy"), self._X_mean)
        np.save(os.path.join(path,name,"_X_std.npy"), self._X_std)
        np.save(os.path.join(path,name,"_y_mean.npy"), self._y_mean)
        np.save(os.path.join(path,name,"_y_std.npy"), self._y_std)
        
        with open(os.path.join(path,name,"_seed.pkl"), "wb") as fp:   #Pickling
            pickle.dump(self._seed, fp)
        with open(os.path.join(path,name,"_train_part.pkl"), "wb") as fp:   #Pickling
            pickle.dump(self._train_part, fp)
        with open(os.path.join(path,name,"_test_part.pkl"), "wb") as fp:   #Pickling
            pickle.dump(self._test_part, fp)
        with open(os.path.join(path,name,"_columns.pkl"), "wb") as fp:   #Pickling
            pickle.dump(self._columns, fp)


    def validateColumn(self, column, mode, dataset = "X", remove_nan=False, **kwargs):
        """
        
        # TODO rewrite
        # Examples + variables
        
        Performs a QC on a column
        =========================
        
        :Example:
            
        ### Quantitative columns
        
        # Will check if column is numerical
        >>> my_dataset.validateColumn("AGE", mode = "quantitative")
        # Will check if column is numerical and remove samples with AGE == np.nan
        >>> my_dataset.validateColumn("AGE", mode = "quantitative", remove_nan = True)
        # Will check if column is numerical and set to np.nan AGE for samples with current AGE Z-score higher than +3 or lower than -3
        >>> my_dataset.validateColumn("AGE", mode = "quantitative", outlier_method = "sd", outlier_sd = 3)
        # Will check if column is numerical and set to np.nan AGE for samples with current AGE lower than 20 or higher than 90
        >>> my_dataset.validateColumn("AGE", mode = "quantitative", outlier_method = "range", outlier_range_low = 20, outlier_range_high = 90)
        
        ### Binary columns
        
        # Will check if column is binary (unique values shape == 2), and convert values to 0/1
        >>> my_dataset.validateColumn("GROUP", mode = "binary")
        # Will check if column is binary and set to np.nan GROUP values which are not "Case" and "Control", and convert values to 0/1
        >>> my_dataset.validateColumn("GROUP", mode = "binary", keys = ["Case", "Control"])
        # Will check if column is binary and set to np.nan GROUP values which are not "Case" and "Control", and convert "Case" to 1 and "Control" to 0 in the GROUP column
        >>> my_dataset.validateColumn("GROUP", mode = "binary", keys = {"Case":1, "Control":0})
        # We can also assign multiple different labels to the same binary values :
        >>> my_dataset.validateColumn("GROUP", mode = "binary", keys = {"Case":1, "Control":0, "Dead":0})

        ### Categorical columns
        # Will one-hot encode column
        >>> my_dataset.validateColumn("TREATMENT", mode = "categorical")
        # Will one-hot encode column and set to np.nan all TREATMENT values not included in the `keys` dictionary
        >>> my_dataset.validateColumn("TREATMENT", mode = "categorical", keys = ["A","B","C"])
        # Will one-hot encode column after modifying values, and set to np.nan all TREATMENT values not included in the `keys` dictionary
        >>> my_dataset.validateColumn("TREATMENT", mode = "categorical", keys = ["A","B","C"])

        Description
        -----------
        
        Performs a quality check with or without modifications on the selected `column`, according to defined parameters.
        All results and modifications if any will be output to the console in order to allow efficient monitoring of the data.
        Dataset must not be frost.
        
        Arguments
        ---------
        
        column : str
            the column to check
            
        mode : str
            either "quantitative" or "qty", "categorical" or "cat" or "binary" or "bin".
            In "quantitative" mode, checks if values are numerical and converts the column to float if not.
            In "categorical" mode, one-hot encodes values.
            In "binary" mode, checks if there are more than 2 unique values, discards outliers if any, and converts to a 0/1 numerical encoding (even if not numerical, or with a different basis (ex: 1/2))
            
        remove_nan: bool
            default False; if True, samples with `np.nan` values for the selected column will be discarded
            
        outlier_method : str, optional
            relevant for "quantitative" mode only. If defined, values in the column will be checked based on standard-deviation (`outlier_method`=="sd") or based on the defined range (`outlier_method`=="range"), and samples with values outside this range will be discarded
            
        outlier_sd : float, optional
            relevant for "quantitative" mode with `outlier_mehod`=="sd" only. If defined, all samples with a Z-score for the selected column higher or lower than the defined standard-deviation will be discarded
            
        outlier_range_low : float, optional
            relevant for "quantitative" mode with `outlier_method`=="range" only. Defines the lower limit for quantitatives values tolerated
        
        outlier_range_high : float, optional
            relevant for "quantitative" mode with `outlier_method`=="range" only. Defines the upper limit for quantitatives values tolerated
        
        keys : list or dict, optional
            relevant for "binary" or "categorical" modes only. Defines the allowed values for the selected column. All samples with other values will be discarded. If a dict is passed, values matching the keys will be replaced by paired values before checking the column. Multiple keys may be replaced by the same value, for binary variables for instance.
        
        min_key_frequency : float, optional
            relevant for "categorical" only. All samples with a value in the selected column with frequency lower than `min_key_frequency` will be discarded
        
        """
        if self._frozen:
            raise Exception("Dataset already frozen, no modifications allowed")
        if mode in ("qty","quantitative",):
            mode = 0
        elif mode in ("bin","binary",):
            mode = 1
        elif mode in ("cat","categorical",):
            mode = 2
        else:
            raise Exception("Mode not understood: {}".format(mode))
        if dataset not in ("X","y",):
            raise Exception("Dataset type not understood: {}".format(dataset))
        df = self._proc_data.copy()
        if column not in df.columns:
            raise Exception("Unknown column: {}".format(column))
        # check for possible errors
        print("Checking column '{}' integrity, with mode='{}'...".format(column,["quantitative","binary","categorical"][mode]))

        if np.sum(pd.isna(df[column]))==0:
            print("  .. Column initially contains no missing values")
        elif np.sum(pd.isna(df[column]))==df.shape[1]:
            raise Exception("Only NA values for column: '{}'".format(column))
        else:
            print("  !! Column initially contains {} missing values".format(np.sum(pd.isna(df[column]))))
            
        def printInvalidValues(a, prefix):
            #column = "TABAC__NON_FUMEUR0_FUMEUR1_SEV"
            #invalid_values = pd.isna(df.loc[:,column]) | ~df.loc[:,column].isin((0,1,))
            #a = df.loc[invalid_values,column]
            outlier_values, outlier_counts = np.unique(a, return_counts=True)
            outlier_values = ["'{}'".format(c) for c in outlier_values.tolist()]
            outlier_counts = ["{}".format(c) for c in outlier_counts.tolist()]
            col_space = max([len(c) for c in outlier_values]) + 2
            print(prefix+"Outlier: "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in outlier_values]))
            print(prefix+"Count:   "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in ["{}".format(f) for f in outlier_counts]]))
            
        #def printInvalidValues(a, prefix):
        #    #column = "TABAC__NON_FUMEUR0_FUMEUR1_SEV"
        #    #invalid_values = pd.isna(df.loc[:,column]) | ~df.loc[:,column].isin((0,1,))
        #    #a = df.loc[invalid_values,column]
        #    na_count = np.sum(np.isnan(a))
        #    outlier_values, outlier_counts = np.unique(a[np.isnan(a)==False], return_counts=True)
        #    outlier_values = np.concatenate((outlier_values, ["nan",])).tolist()
        #    outlier_counts = np.concatenate((outlier_counts, [na_count,])).tolist()
        #    col_space = max([len(c) for c in outlier_values]) + 5
        #    print(prefix+"Outlier: "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in outlier_values]))
        #    print(prefix+"Count:   "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in ["{}".format(f) for f in outlier_counts]]))
            
        if mode==0:
            # check if column is already type float or int
            if df.dtypes[column] in [float, int, "float32", "float64", "int32", "int64"]:
                print("  .. Column is already in numeric format")
            else:
                print("  !! Column is not in numeric format")
                temp_column=df[column].astype(str)
                # check by regex
                invalid_values = (~pd.isna(df[column])) & (~temp_column.str.match("^[0-9.-]+$"))
                if np.sum(invalid_values)>0: # non-numeric values found
                    printInvalidValues(df.loc[invalid_values,column], "  !!!!!! ")
                    print("  !!!!!! Found {} invalid values, setting to nan".format(np.sum(invalid_values)))
                    temp_column[invalid_values]=np.nan
                else:
                    print("  .... No invalid values found, unchanged")
                # convert to numeric
                temp_column[pd.isna(df[column])]=np.nan
                df[column] = pd.to_numeric(temp_column, errors="coerce")
                print("  .... Converted to numeric, now contains {} missing values".format(np.sum(pd.isna(df[column]))))
            # check outliers if asked
            if 'outlier_method' in kwargs.keys() and kwargs['outlier_method'] is not None:
                if kwargs['outlier_method'] == 'sd':
                    mean_value = np.nanmean(df[column])
                    sd_value = np.std(df[column])
                    lo_threshold = mean_value-kwargs['outlier_sd']*sd_value
                    hi_threshold = mean_value+kwargs['outlier_sd']*sd_value
                    print("  .. Checking outliers based on sd: {} and range: [{:2f},{:2f}]".format(kwargs['outlier_sd'],lo_threshold,hi_threshold))
                    outlier_values = (~pd.isna(df[column]) & (df[column]<lo_threshold) | (df[column]>hi_threshold))
                    if np.sum(outlier_values)>0:
                        print("  !!!!!! Found {} outliers, setting to nan".format(np.sum(outlier_values)))
                        printInvalidValues(df.loc[outlier_values,column], "  !!!!!! ")
                        df.loc[outlier_values,column] = np.nan
                    else:
                        print("  ...... No outliers found, unchanged")
                elif kwargs['outlier_method'] == 'range':
                    lo_threshold = kwargs['outlier_range_low']
                    hi_threshold = kwargs['outlier_range_high']
                    print("  .. Checking outliers based on range: [{},{}]".format(lo_threshold,hi_threshold))
                    outlier_values = (~pd.isna(df[column]) & (df[column]<lo_threshold) | (df[column]>hi_threshold))
                    if np.sum(outlier_values)>0:
                        print("  !!!!!! Found {} outliers, setting to nan".format(np.sum(outlier_values)))
                        printInvalidValues(df.loc[outlier_values,column], "  !!!!!! ")
                        df.loc[outlier_values,column] = np.nan
                    else:
                        print("  ...... No outliers found, unchanged")
            # remove nan if asked
            if remove_nan:
                print("  !! Removing {} missing values".format(np.sum(pd.isna(df[column]))))
                df = df.loc[~pd.isna(df[column])]
            else:
                print("  .. {} missing values found, unchanged".format(np.sum(pd.isna(df[column]))))
            self._columns.append(dict(colname=column, mode=0, dataset=dataset, group=len(self._columns)))
            
            self._proc_data = df
            return column

        elif mode==1:
            # IF KEYS DICTIONARY :
            # # check that dictionary is valid : only two possible values, 0 and 1 for final variable
            # # set var dictionary according to dict passed as parameter, other values as na.
            # ELSE (NO KEYS DICTIONARY) :
            # # check if valid keys list passed :
            # # # if Yes : check if valid (2 values), and set all other values to nan
            # # # if No : create the list by unique size
            # # : convert to 0/1 and make sure variable is numeric
            keys = None
            if 'keys' in kwargs.keys() and kwargs['keys'] is not None:
                keys = kwargs["keys"]
            if isinstance(keys, dict):
                possible_final_values = np.unique(list(keys.values())).tolist()
                if len(possible_final_values)==2 and (possible_final_values == [0,1] or possible_final_values == [1,0]):
                    pass
                else:
                    raise Exception("Possible values for binary variable in key dictionary must be 0 and 1")
                # replace values
                print("  !! Setting new column binary dictionary to: {}".format(keys))
                outlier_values = (~df[column].isin(keys.keys())) & (~pd.isna(df[column])) # find values != nan and not in keys
                if np.sum(outlier_values)>0:
                    print("  !!!!!! Found {} unvalid values, setting to nan".format(np.sum(outlier_values)))
                    printInvalidValues(df.loc[outlier_values,column], "  !!!!!! ")
                    df.loc[outlier_values,column] = np.nan # values which are not in the keys are set to np.nan
                df[column] = df[column].replace(keys) # values which are in the keys are replaced by their intended final value
                # that's it
            elif isinstance(keys, list):
                if len(keys)!=2:
                    raise Exception("A binary variable must have exactly 2 possible values")
                outlier_values = (~pd.isna(df[column])) & (~df[column].isin(keys)) # select not na values not in wanted keys
                if np.sum(outlier_values)>0:
                    print("  !!!!!! Found {} unvalid values, setting to nan".format(np.sum(outlier_values)))
                    printInvalidValues(df.loc[outlier_values,column], "  !!!!!! ")
                    df.loc[outlier_values,column] = np.nan
                else:
                    print("  ...... No unvalid values found, unchanged")
            else:
                # check if only two possible values
                if np.unique(df.loc[~pd.isna(df[column]),column]).shape[0]==1:
                    raise Exception("Column: {} > only one possible value found in binary column?".format(column))
                elif np.unique(df.loc[~pd.isna(df[column]),column]).shape[0]==2:
                    print("  .. Found only binary values: {}".format(np.unique(df[column])))
                else:
                    # create our own dictionary
                    keys = np.unique(df[column],return_counts=True)
                    keys = keys[0][np.argsort(-keys[1])][:2]
                    if np.all(keys == [1,0]):
                        keys=np.array([0,1])
                    print("  .. More than two possible values found, checking outliers based on most frequent keys: {}".format(keys))
                    outlier_values = (~pd.isna(df[column])) & (~df[column].isin(keys)) # select not na values not in wanted keys
                    if np.sum(outlier_values)>0:
                        print("  !!!!!! Found {} unvalid values, setting to nan".format(np.sum(outlier_values)))
                        printInvalidValues(df.loc[outlier_values,column], "  !!!!!! ")
                        df.loc[outlier_values,column] = np.nan
                    else:
                        print("  ...... No unvalid values found, unchanged")
                            
            # set values to 0/1 instead of whatever they could be
            if np.all(df[column].isin([0,1,np.nan])):
                print("  .. Column has correct binary dictionary (0, 1), unchanged")
            else:
                raw_keys = np.unique(df.loc[~pd.isna(df[column]),column])
                new_keys_dict = dict(zip(raw_keys, [0,1]))
                print("  !! Setting new column binary dictionary to: {}".format(new_keys_dict))
                df[column] = df[column].replace(new_keys_dict)
            # convert to numeric
            df[column] = pd.to_numeric(df[column])
            # remove nan if asked
            if remove_nan:
                print("  !! Removing {} missing values".format(np.sum(pd.isna(df[column]))))
                df = df.loc[~pd.isna(df[column])]
            else:
                print("  .. {} missing values found, unchanged".format(np.sum(pd.isna(df[column]))))
            print("  .. Positive values: {:.1f}%".format(100*np.sum(df[column]==1)/df.shape[0]))
            print("  .. Negative values: {:.1f}%".format(100*np.sum(df[column]==0)/df.shape[0]))
            self._columns.append(dict(colname=column, mode=1, dataset=dataset, group=len(self._columns)))

            self._proc_data = df
            return column
        
        elif mode==2:
            # copy column
            ncol = df[column].copy()
            # categorical :
            if "keys" in kwargs.keys() and kwargs['keys'] is not None:
                keys = kwargs["keys"]
                #keys = dict(lo="lo", mid="lo", hi="hi", extrahi="hi", AWESOME="hi")
                if isinstance(keys, dict):
                    ckeys = list(keys.keys())
                    keys_not_found = [key for key in ckeys if np.sum(ncol==key)==0]
                    if len(keys_not_found) > 0:
                        print("  !! Some keys were not found in the column: {}".format(keys_not_found))
                        print("  !! Those keys will be ignored")
                        # remove those keys
                        ckeys = [key for key in ckeys if key not in keys_not_found]
                    keys = {key: keys[key] for key in ckeys}
                    # replace
                    ncol = ncol.replace(keys)
                    # and keep trace of desired keys
                    keys = list(keys.values())
                    # make sure no duplicates
                    keys = np.unique(keys).tolist()
                elif isinstance(keys, list):
                    keys_not_found = [key for key in keys if np.sum(ncol==key)==0]
                    if len(keys_not_found) > 0:
                        print("  !! Some keys were not found in the column: {}".format(keys_not_found))
                        print("  !! Those keys will be ignored")
                        # remove those keys
                        keys = [key for key in keys if key not in keys_not_found]
                else:
                    raise Exception("Argument keys passed as parameter must be either a dict or a list")
                # check if all keys are represented
                # if not, remove those keys
            else:
                keys = np.unique(ncol).tolist()
            # check frequency for each key:
            key_print = ["{}".format(key) for key in keys]
            key_count = [np.sum(ncol==key) for key in keys]
            key_freq = key_count/np.sum(key_count)
            col_space = max(7,max([len(key) for key in key_print])+2)
            print("  .. Keys count/frequency:")
            print("       Key:   "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in key_print]))
            print("       Count: "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in ["{}".format(f) for f in key_count]]))
            print("       Freq:  "+"".join(["".join((" ",) * (col_space-len(key)))+key for key in ["{:.1f}%".format(100*f) for f in key_freq]]))
            # check if desired min key freq/occ
            min_key_count = 1
            min_key_freq = .05
            if "min_key_count" in kwargs.keys():
                min_key_count = kwargs["min_key_count"]
            if "min_key_freq" in kwargs.keys():
                min_key_freq = kwargs["min_key_freq"]
            # compare minimums to observed
            under_min_count = [key for key, cnt in zip(keys, key_count) if cnt < min_key_count]
            under_min_freq = [key for key, frq in zip(keys, key_freq) if frq < min_key_freq]
            if len(under_min_count)>0:
                print("  !! Keys have count lower than threshold: {}".format(under_min_count))
                print("  !! Values matching those keys will be set to nan")
                ncol.loc[ncol.isin(under_min_count)] = np.nan
                keys = [key for key in keys if key not in under_min_count]
            if len(under_min_freq)>0:
                print("  !! Keys have freq lower than threshold: {}".format(under_min_freq))
                print("  !! Values matching those keys will be set to nan")
                ncol.loc[ncol.isin(under_min_freq)] = np.nan
                keys = [key for key in keys if key not in under_min_freq]
                
            # actually remove values which are not listed in keys
            outlier_values = ~ncol.isin(keys) & ~pd.isna(ncol)
            if np.sum(outlier_values)>0:
                print("  !! Found {} unvalid values, setting to nan".format(np.sum(outlier_values)))
                printInvalidValues(ncol.loc[outlier_values], "  !!!!!! ")
                ncol.loc[outlier_values] = np.nan
            else:
                print("  .. No unvalid values found, unchanged")
            # now convert
            multi_label_separator = None
            allow_multi_label = False # default = False
            if "multi_label_separator" in kwargs.keys():
                multi_label_separator = kwargs["multi_label_separator"]
            if "allow_multi_label" in kwargs.keys():
                allow_multi_label = kwargs["allow_multi_label"]

            if allow_multi_label:
                # ohe hot encode
                ohe_df = ncol.str.get_dummies(sep = multi_label_separator)
                # rename variables
                ohe_df.columns = ncol.name+"_"+ohe_df.columns
                # add variables where the initial variable was
                col_index = np.where(df.columns==column)[0][0]
                df = pd.concat((df.iloc[:,:col_index],ohe_df,df.iloc[:,(col_index+1):]), axis=1)
                for c in ohe_df.columns.tolist():
                    self._columns.append(dict(colname=c, mode=1, dataset=dataset, group=len(self._columns)))
                    
                self._proc_data = df
                return ohe_df.columns.tolist()
            else:
                # one hot encode
                ohe_df = pd.get_dummies(ncol, prefix=ncol.name)
                # adjust column names if float (due to np.na values)
                ohe_df.columns = [c[:-2] if c[-2:]==".0" else c for c in ohe_df.columns.tolist()]
                # add variables where the initial variable was
                col_index = np.where(df.columns==column)[0][0]
                df = pd.concat((df.iloc[:,:col_index],ohe_df,df.iloc[:,(col_index+1):]), axis=1)
                group =len(self._columns)
                for c in ohe_df.columns.tolist():
                    self._columns.append(dict(colname=c, mode=1, dataset=dataset, group=group))
                
                self._proc_data = df
                return ohe_df.columns.tolist()

        
    def ImputeMissingValues(self, **kwargs):
        """
        
        # TODO rewrite help
        
        Performs missing values imputation
        =========================
        
        :Example:
            
        >>> my_dataset.ImputeMissingValues()
        >>> my_dataset.ImputeMissingValues(columns = ["DRUG1","DRUG2","TREATMENT"])

        Description
        -----------
        
        Performs missing values imputations based on either KNN, Autoencoder, or best method between both.
        If `method` is set to "auto", an autoencoder and a KNN will be trained to impute randomly chosen missing values,
        and the best model will be selected based on loss on a test set, and retrained on the full dataset to impute missing values
        
        Arguments
        ---------
        
        columns : list, optional, default=None
            list of quantitative columns to include in the missing values imputation
            
        """
        if self._frozen:
            raise Exception("Dataset already frozen, no modifications allowed")
            
        # we should check it there are any duplicated columns in the arguments
        if "columns" in kwargs.keys():
            all_columns = [c["colname"] for c in self._columns]
            tmp = kwargs["columns"]
            # check if columns have been validated
            flag = 0
            for c in tmp:
                if c not in all_columns:
                    flag = 1
                    print("Unknown or unvalidated column: '{}'".format(c))
            if flag>0:
                raise Exception("Unknown or unvalidated columns, aborting")
            if np.unique(tmp).shape[0] != len(tmp):
                raise Exception("Duplicated columns, aborting")
            # check that if a column has been passed as a parameter, all columns for the same group are passed too
            for column in tmp:
                group = [c["group"] for c in self._columns if c["colname"]==column][0]
                columns_in_group = [c["colname"] for c in self._columns if c["group"]==group]
                if all([c in tmp for c in columns_in_group]) != True:
                    raise Exception("All one-hot encoded mono-label variables of the same group must be included (column: '{}'".format(column))
            
            columns = tmp
        else:
            # by default : take all columns which are in the X dataset
            columns = [c["colname"] for c in self._columns if c["dataset"]=="X"]
        
        df = self._proc_data.copy()
        
        # Remove columns with only missing values
        cols_to_remove = []
        for coln in columns:
            if np.sum(pd.isna(df[coln]))==df.shape[0]:
                print("    !!!! Column {} (qty) only has missing values, will be removed".format(coln))
                cols_to_remove.append(coln)
        # Check if 0 variance => would lead to NA values
        for coln in columns:
            if np.nanvar(df[coln])==0:
                print("    !!!! Column {} (qty) has 0 variance, will be removed".format(coln))
                cols_to_remove.append(coln)
        if len(cols_to_remove)>0:
            columns = [c for c in columns if c not in cols_to_remove]                        
        df = df.loc[:,columns]
        mean = np.nanmean(df,axis=0)
        std = np.nanstd(df,axis=0)
        df = (df-mean)/std
                
        print("    >>>> Total missing values:\n{}".format(np.sum(pd.isna(df))))
            
        ar = np.array(df)
        
        # impute
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed_np = imputer.fit_transform(ar)
        
        # unnormalize
        unnorm_imputed_np = imputed_np*std+mean
        unnorm_imputed_df = df.copy()
        # send back to dataframe, with same structure: index, columns...
        unnorm_imputed_df.loc[:,:] = unnorm_imputed_np
        
        # for binary variables : convert to 0/1 and ensure that variables in the same group have only one 1
        # convert by group to make it easier
        # one entry per group
        bin_columns_by_group = {}
        for c in self._columns:
            if c["mode"]==1 and c["colname"] in columns:
                if c["group"] in bin_columns_by_group.keys():
                    bin_columns_by_group[c["group"]].append(c["colname"])
                else:
                    bin_columns_by_group[c["group"]]=[c["colname"]]
        
        for cols in bin_columns_by_group.values():
            if len(cols)==1: # binary
                unnorm_imputed_df[cols[0]] = np.minimum(1, np.maximum(0, np.round(unnorm_imputed_df[cols[0]])))
            else:
                a = np.argmax(np.array(unnorm_imputed_df[cols]), axis=1)
                b = np.zeros((a.size, a.max()+1))
                b[np.arange(a.size),a] = 1
                unnorm_imputed_df[cols] = b
        
        self._proc_data.loc[:,columns] = unnorm_imputed_df
        
        
    def extractResiduals(self, align_column, ctrl_columns, plot_path=None, color_binary_column = None):
        from sklearn.linear_model import LinearRegression

        df = self._proc_data
        ctrl_df = df.copy()
        
        for clmn in ctrl_columns:
            # compute pearson correlation coefficient
            pearson_r = df[[align_column,clmn]].corr().iloc[0,1]
            # compute linear regression between column over which to align data, and column to be controlled
            mod = LinearRegression()
            mod.fit(X = np.array(df[align_column]).reshape(-1,1), y = np.array(df[clmn]).reshape(-1,1))
            # predict according to align variable
            predicted_according_to_align_column = mod.predict(X = np.array(df[align_column]).reshape(-1,1)).reshape(-1)
            # compute residuals by removing predicted value
            residuals = np.array(df[clmn]) - predicted_according_to_align_column
            # replace
            ctrl_df[clmn] = residuals
            
            print("Removed variance bound to {} from var: {} (r={:.3f})".format(align_column, clmn, pearson_r))
            
            if plot_path is not None and isinstance(plot_path, str):
                from matplotlib import pyplot as plt

                # recompute linear regression versus residuals
                mod2 = LinearRegression()
                mod2.fit(X = np.array(df[align_column]).reshape(-1,1), y = np.array(ctrl_df[clmn]).reshape(-1,1))
                new_predicted_according_to_align_column = mod2.predict(X = np.array(df[align_column]).reshape(-1,1)).reshape(-1)
                # plot
                plt.figure(figsize=(12,6))
                # initial data
                plt.subplot(1, 2, 1)
                if color_binary_column is not None:
                    plt.scatter(df.loc[df[color_binary_column]==1,align_column], df.loc[df[color_binary_column]==1,clmn], c = "r")
                    plt.scatter(df.loc[df[color_binary_column]==0,align_column], df.loc[df[color_binary_column]==0,clmn], c = "b")
                else:
                    plt.scatter(df[align_column], df[clmn], c = "r")
                plt.ylabel("{}".format(clmn))
                plt.xlabel("{}".format(align_column))
                plt.plot(df[align_column], predicted_according_to_align_column, c = "black")
                plt.title("Initial data (r={:.3f})".format(pearson_r))
                # residuals
                plt.subplot(1, 2, 2)
                if color_binary_column is not None:
                    plt.scatter(df.loc[df[color_binary_column]==1,align_column], ctrl_df.loc[df[color_binary_column]==1,clmn], c = "r")
                    plt.scatter(df.loc[df[color_binary_column]==0,align_column], ctrl_df.loc[df[color_binary_column]==0,clmn], c = "b")
                else:
                    plt.scatter(df[align_column], ctrl_df[clmn], c = "r")
                plt.ylabel("{}".format(clmn))
                plt.xlabel("{}".format(align_column))
                plt.plot(df[align_column], new_predicted_according_to_align_column, c = "black")
                plt.title("Residuals")
                plt.savefig(os.path.join(plot_path,"residuals_{}.png".format(clmn)))
            
        self._proc_data = ctrl_df

        
    def freezeDataset(self, standardize = "all",
                      test_partition_size_pct=.2, stratified_partitionning_column=None):
        """
        
        Freezes dataset, allowing analysis
        ==================================
        
        :Example:
            
        >>> my_dataset.freezeDataset(standardize = "X",
                                     stratified_partitionning_column = "GROUP",
                                     test_partition_size_pct = .2,
                                     seed = 846)
        
        Description
        -----------
        
        Convert the dataframe with all modifications performed into X and y datasets,
        partitionned in training and test sets according to the `test_partition_size_pct` factor
        
        Arguments
        ---------
        
        standardize : optional, default = "all"
            whether to standardize (variance=1, mean=0) values in the X dataset.
            can be either : 
                str
                    "all": standardize all variables
                    "none": standardize no variables
                    "X": standardize all X variables
                    "y": standardize all y variables
                a list of column names to standardize

        test_partition_size_pct : float, default=0.2
            the test set size partitioning factor (0 to 1)
            
        stratified_partitionning_column : str or pandas.Series, optional, default=None
            column to use for stratified partitioning, if specified. If passed as str, will be taken from the current dataset.
            
        seed : int, optional
            the seed for partitioning

        """
        if self._frozen:
            raise Exception("Dataset already frozen, no modifications allowed")
            
        # retrieve X columns and y columns
        X_columns = [c["colname"] for c in self._columns if c["dataset"]=="X"]
        y_columns = [c["colname"] for c in self._columns if c["dataset"]=="y"]
        if len(X_columns)==0:
            raise Exception("No columns in X dataset !")
        if len(y_columns)==0:
            raise Exception("No columns in y dataset !")
            
        # we should check if there are any duplicates in the filter columns list
        if np.unique(X_columns).shape[0] != np.array(X_columns).shape[0]:
            raise Exception("Duplicates in X_columns, aborting")
        if np.unique(y_columns).shape[0] != np.array(y_columns).shape[0]:
            raise Exception("Duplicates in y_columns, aborting")
        
        if standardize == "all":
            standardize = X_columns.copy()
            standardize.extend(y_columns)
        elif standardize == "none":
            standardize = []
        elif standardize == "X":
            standardize = X_columns.copy()
        elif standardize == "y":
            standardize = y_columns.copy()
        else:
            if np.unique(standardize).shape[0] != np.array(standardize).shape[0]:
                raise Exception("Duplicates in onehotencode, aborting")
            standardize=np.array(standardize).reshape(-1).tolist()
            
        print("Freezing dataset...")
        
        df = self._proc_data.copy()
        
        # We can partition our datasets
        y_strat = None
        if stratified_partitionning_column is not None:
            if isinstance(stratified_partitionning_column, pd.Series) or isinstance(stratified_partitionning_column, np.ndarray):
                y_strat = stratified_partitionning_column
            else:
                y_strat = df[stratified_partitionning_column]
        # partition
        train_part, test_part = train_test_split(np.arange(df.shape[0]), stratify = y_strat, test_size=test_partition_size_pct, random_state=self._seed)
        
        # get X df
        X_df = df.loc[:,X_columns]
                
        # initially, compute mean and std for all variables
        X_mean = np.array(np.mean(X_df.iloc[train_part],axis=0))
        X_std = np.array(np.std(X_df.iloc[train_part],axis=0))
        # for variables for which we do not want standardization, we'll set mean and std to 0
        # set mean to 0 and std to 1 for variables we don't want to standardize
        X_mean[[coln not in standardize for coln in X_df.columns]] = 0
        X_std[[coln not in standardize for coln in X_df.columns]] = 1
        
        X_train = np.array((X_df.iloc[train_part]-X_mean)/X_std)
        X_test = np.array((X_df.iloc[test_part]-X_mean)/X_std)
        
        # get y df
        y_df = df.loc[:,y_columns]

        # initially, compute mean and std for all variables
        y_mean = np.array(np.mean(y_df.iloc[train_part],axis=0))
        y_std = np.array(np.std(y_df.iloc[train_part],axis=0))
        # for variables for which we do not want standardization, we'll set mean and std to 0
        # set mean to 0 and std to 1 for variables we don't want to standardize
        y_mean[[coln not in standardize for coln in y_df.columns]] = 0
        y_std[[coln not in standardize for coln in y_df.columns]] = 1
        
        y_train = np.array((y_df.iloc[train_part]-y_mean)/y_std)
        y_test = np.array((y_df.iloc[test_part]-y_mean)/y_std)
                
        # Check standardization errors
        X_std_errs = np.where(X_std==0)[0]
        for err in X_std_errs:
            print("!! Warning: X column {} ({}) has 0 variance in training set".format(err, X_df.columns[err]))
        y_std_errs = np.where(y_std==0)[0]
        for err in y_std_errs:
            print("!! Warning: y column {} ({}) has 0 variance in training set".format(err, y_df.columns[err]))
        if len(X_std_errs)+len(y_std_errs) > 0:
            raise Exception("0 variance in training data. Please remove variables with 0 variance or try changing the seed or partitionning rate.")
            
        # save all
        self._train_part = train_part
        self._test_part = test_part
        
        self._X_mean = X_mean
        self._X_std = X_std
        self._y_mean = y_mean
        self._y_std = y_std
        
        self._X_train = X_train
        self._X_test = X_test

        self._y_train = y_train
        self._y_test = y_test

        self._frozen=True
        
        print("Dataset frost\n  .. Training set -X: {} -y: {}\n  .. Test set     -X: {} -y: {}".format(self._X_train.shape,self._y_train.shape,self._X_test.shape,self._y_test.shape))

    def getXmeanstd(self):
        """
        
        Get mean and std of each variable
        =================================
        
        :Example:
            
        X_mean, X_std = my_dataset.getXmeanstd()
        
        Description
        -----------
        
        Returns both X mean and std for each column, before normalization. DatasetManager instance must be frozen.

        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        return (self._X_mean,self._X_std)

    def getYmeanstd(self):
        """
        
        Get mean and std of each variable
        =================================
        
        :Example:
            
        y_mean, y_std = my_dataset.getYmeanstd()
        
        Description
        -----------
        
        Returns both y mean and std for each column, before normalization. DatasetManager instance must be frozen.

        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        return (self._y_mean,self._y_std)
        
    def getTrainingSet(self, unstandardize = False):
        """
        
        Get training set
        ================
        
        :Example:
            
        >>> X_train, y_train = my_dataset.getTrainingSet()

        Description
        -----------
        
        Returns both X and y training sets. DatasetManager instance must be frozen.

        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        if unstandardize:
            return (self._X_train*self._X_std+self._X_mean,
                    self._y_train*self._y_std+self._y_mean)
        return (self._X_train,self._y_train)
        
    def getTestSet(self, unstandardize = False):
        """
        
        Get test set
        ============
        
        :Example:
            
        >>> X_test, y_test = my_dataset.getTestSet()

        Description
        -----------
        
        Returns both X and y test sets. DatasetManager instance must be frozen.

        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        if unstandardize:
            return (self._X_test*self._X_std+self._X_mean,
                    self._y_test*self._y_std+self._y_mean)
        return (self._X_test,self._y_test)

    def getFullSet(self, unstandardize = False):
        """
        
        Get test set
        ============
        
        :Example:
            
        >>> X, y = my_dataset.getFullSet()

        Description
        -----------
        
        Returns both X and y concatenated trainign and test sets. DatasetManager instance must be frozen.

        """
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        X_full = np.concatenate([self._X_train, self._X_test], axis=0)
        y_full = np.concatenate([self._y_train, self._y_test], axis=0)
        if unstandardize:
            return (X_full*self._X_std+self._X_mean,
                    y_full*self._y_std+self._y_mean)
        return (X_full,y_full)

    def getColnames(self, dataset="X"):
        """
        
        Get column names for the X (input) or y (output) dataset
        ==========================================
        
        :Example:
            
        >>> X_cols = my_dataset.getColnames("X")
        >>> y_cols = my_dataset.getColnames("y")

        Description
        -----------
        
        Returns column names for the X/y dataset.
        
        Parameters
        ----------
        
        dataset : str, either "X" or "y"

        """
        if dataset in ("X","x",):
            return [c["colname"] for c in self._columns if c["dataset"]=="X"]
        elif dataset in ("y","Y",):
            return [c["colname"] for c in self._columns if c["dataset"]=="y"]
        else:
            raise Exception("Dataset unknown: {}".format(dataset))
    
    def BootStrapOverTrainingSet(self, iterations = 10, partitionning = .8, stratified_partitionning_columns=None, use_smote_on_training = False, verbose=False, replace = True, seed = None):
        """
        
        Creates an iterator that will bootstrap over the traning set
        ============================================================
        
        # TODO rewrite
        
        :Example:
            
        >>> for X_train, y_train, X_val, y_val, train_indices, val_indices, i in my_dataset.BootStrapOverTrainingSet():
        >>>     print("Training on {} samples, validating on {} samples".format(len(train_indices), len(val_indices)))
        >>>     model.fit(X_train, y_train)
        >>>     print("Results for bootstrap no {}:".format(i))
        >>>     print("{}".format(loss(model.predict(X_val, y_val))))

        Description
        -----------
        
        Randomly bootstraps over the training set, creating a new sub-training set and a validation set at each iteration
        
        Parameters
        ----------
        
        iterations : int, default=10
            the number of iterations for the boostrap ; at each iterations, new training and validation sets will be sampled from the initial training set
            
        partitionning : float, default=0.8
            the size, in percent of the initial training set (0 to 1) of samples to include in the new training sets
            
        stratified_partitionning_columns : str, optional, default=None
            the stratified column if desired
            
        training_smote : bool, optional, default=False
            if set to True, SMOTE will be applied on each training set
            
        seed : int, optional, default=None
            optional seed used for SMOTE

        """
        from sklearn.utils import resample
        
        if not self._frozen:
            raise Exception("Dataset must be frozen")
        
        if seed is None:
            seed = self._seed
            
        X, y = self.getTrainingSet()
            
        if stratified_partitionning_columns is not None:
            # TODO we still use _proc_data for stratifying !! Should not be done this way ?
            if isinstance(stratified_partitionning_columns, list):
                y_strat = self._proc_data[stratified_partitionning_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            else:
                y_strat = self._proc_data[stratified_partitionning_columns]
            y_strat = np.array(y_strat.iloc[self._train_part])
        else:
            y_strat = None
        
        for i in range(iterations):
            bootstrap_train = resample(np.arange(X.shape[0]),
                                       replace=replace,
                                       n_samples=np.round(partitionning*X.shape[0]).astype(int),
                                       stratify = y_strat,
                                       random_state = self._seed+i) # seed + i -> same each time we call the function with same seed & number of iterations
            bootstrap_test = np.setdiff1d(np.arange(X.shape[0]), bootstrap_train)
            
            X_bootstrap_train = X[bootstrap_train,:]
            y_bootstrap_train = y[bootstrap_train,:]
            X_bootstrap_test = X[bootstrap_test,:]
            y_bootstrap_test = y[bootstrap_test,:]
            
            if use_smote_on_training:
                from imblearn.over_sampling import SMOTE

                sm = SMOTE(random_state = self._seed+i) # same seed as bootstrap resampler
                X_bootstrap_train, y_bootstrap_train = sm.fit_resample(X_bootstrap_train, y_bootstrap_train)
                if verbose:
                    print("Returning bootstrap with training set size: {} ({} before SMOTE), validation set size: {}".format(X_bootstrap_train.shape[0],bootstrap_train.shape[0],bootstrap_test.shape[0]))
            elif verbose:
                print("Returning bootstrap with training set size: {}, validation set size: {}".format(bootstrap_train.shape[0],bootstrap_test.shape[0]))
                
            yield (X_bootstrap_train,y_bootstrap_train,
                   X_bootstrap_test, y_bootstrap_test,
                   bootstrap_train, bootstrap_test,
                   i)
            
            
# %%
            
if __name__ == "__main__":
    debug_df = pd.DataFrame({'qtY1': [168, 175, 182, 156, 160, 184, 189, 172, 163, 149], # quantitative distrib, range = 100-300, no abnormalities
                             'qty2': [168, np.nan, 182, 156, 160, np.nan, 189, 172, 163, 149], # quantitative distrib, range = 100-300, NA values
                             'Qté 3': [168, 175, 3, 156, 160, 184, 189, 172, 163, 999], # quantitative distrib, range = 100-300, lo and hi outliers
                             'qty=4.': [1.2, -1.4, .2, -.8, -2.9, -3.1, 1.4, 1.8, -1.7, -.2], # quantitative distrib, range = -3;3, no outliers
                             'qty5': [1.2, -1.4, .2, -.8, -2.9, -15.1, 1.4, 18.8, -1.7, -.2], # quantitative distrib, range = -3;3, outliers
                             'qty6': ['1.2', -1.4, '.2', '-.8', -2.9, -3.1, '1.4', '1.8', -1.7, -.2], # quantitative distrib, range = -3;3, no outliers, textual values
                             'qty7 reeaaaaally looooong NAAME!!! (like, really, really long)': ['1.2', '-1.4', '.2', '-.8', '-2.9', '-3.1', '1.4', '1.8', '-1.7', '-.2'], # quantitative distrib, range = -3;3, no outliers, textual values
                             'qty8': ['1.2', 'ND', '.2', '-.8', '-2.9', '-3.1', '1.4', ' ', '-1.7', '-.2'], # quantitative distrib, range = -3;3, no outliers, textual values + ND value
                             'qty   9!': ['1.2', -1.4, np.nan, '-.8', -2.9, -3.1, '1.4', '1.8', -1.7, -.2], # numeric + text + nan value
                             'qty10': ['1.2', -1.4, np.nan, 'ND', -2.9, -3.1, '1.4', '1.8', -1.7, -.2], # numeric + text + nan value + ND value
                             'bin1': [0,1,0,0,1,1,0,1,0,1],
                             'bin2': [0,np.nan,0,0,1,np.nan,0,1,0,1],
                             'bin3': [4,9,4,4,4,9,9,4,9,9],
                             'bin4': ['jose','garcia','jose','jose','garcia','garcia','jose','garcia','jose','garcia'],
                             'bin5': [0,1,0,3,1,1,0,1,0,1],
                             'cat1': [0,1,3,2,1,0,2,3,2,0],
                             'cat2': [0,1,3,2,1,8,2,3,2,0],
                             'cat3': [4,1,3,2,1,8,2,3,2,4],
                             'cat4': ['lo','mid','hi','mid','mid','lo','hi','hi','lo','mid'],
                             'cat5': ['lo','mid','hi','mid','mid','lo','hi','hi','lo','mid'],
                             'cat6': ['lo','mid','hi','mid','extrahi','lo','hi','hi','lo','mid'],
                             'cat7': ['lo','mid','hi','mid','extrahi','lo','hi','hi','lo','mid'],
                             'y':    [0,0,0,0,0,1,1,1,1,1]})
    debug_df = debug_df.rename(columns=dict(qty6="qty5"))
    debug_df = debug_df.rename(columns=dict(bin3="bin2"))
    
    def compareCols(a, b):
        if isinstance(a, pd.Series):
            a = a.tolist()
        if isinstance(b, pd.Series):
            b = b.tolist()
        if len(a) != len(b):
            return False
        res = [a[i] == b[i] or (pd.isna(a[i]) and pd.isna(b[i])) for i in list(range(len(a)))]
        return all(res)
    
    def argmaxna(a):
        am = np.argmax(a,axis=1).astype(np.float)
        am[np.sum(a,axis=1)==0] = np.nan
        return am

    ##### INI
    
    dset = DatasetManagerV3(debug_df)
    if all(dset._proc_data.columns == ['QTY1', 'QTY2', 'QTE3', 'QTY4', 'QTY5v1', 'QTY5v2',
                                       'QTY7REEAAAAALLYLOOOOONGNAAMELI', 'QTY8', 'QTY9', 'QTY10', 'BIN1',
                                       'BIN2v1', 'BIN2v2', 'BIN4', 'BIN5', 'CAT1', 'CAT2', 'CAT3', 'CAT4',
                                       'CAT5', 'CAT6', 'CAT7', 'Y']) != True:
        raise Exception("Error at initialization")
    else:
        print("Initialization ok")

    ##### QTY

    step = 1
    colnms = ["qtY1","QTY1","QTY1"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = debug_df.loc[:,colnms[0]]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 2
    colnms = ["qty2","QTY2","QTY2"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [168, np.nan, 182, 156, 160, np.nan, 189, 172, 163, 149]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 3
    colnms = ["qté 3","QTE3","QTE3"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty", outlier_method = "range", outlier_range_low = 100, outlier_range_high = 300)
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [168, 175, np.nan, 156, 160, 184, 189, 172, 163, np.nan]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 4
    colnms = ["qty=4.","QTY4","QTY4"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty", outlier_method = "sd", outlier_sd = 3)
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = debug_df.loc[:,colnms[0]]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 5
    colnms = ["qty5","QTY5v1","QTY5v1"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty", outlier_method = "sd", outlier_sd = 1)
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, -1.4, .2, -.8, -2.9, np.nan, 1.4, np.nan, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 6
    colnms = ["qty5","QTY5v2","QTY5v2"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, -1.4, .2, -.8, -2.9, -3.1, 1.4, 1.8, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 7
    colnms = ["qty7 reeaaaaally looooong NAAME!!! (like, really, really long)","QTY7REEAAAAALLYLOOOOONGNAAMELI","QTY7REEAAAAALLYLOOOOONGNAAMELI"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, -1.4, .2, -.8, -2.9, -3.1, 1.4, 1.8, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 8
    colnms = ["qty8","QTY8","QTY8"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, np.nan, .2, -.8, -2.9, -3.1, 1.4, np.nan, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 9
    colnms = ["qty9","QTY9","QTY9"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, -1.4, np.nan, -.8, -2.9, -3.1, 1.4, 1.8, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 10
    colnms = ["qty10","QTY10","QTY10"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="qty")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1.2, -1.4, np.nan, np.nan, -2.9, -3.1, 1.4, 1.8, -1.7, -.2]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    ##### BINARY
        
    step = 11
    colnms = ["bin1","BIN1","BIN1"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = debug_df.loc[:,colnms[0]]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 12
    colnms = ["bin2","BIN2v1","BIN2v1"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = debug_df.loc[:,colnms[0]].iloc[:,0]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 13
    colnms = ["bin3","BIN2v2","BIN2v2"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [0,1,0,0,0,1,1,0,1,1,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 14
    colnms = ["bin3","BIN2v2","BIN2v2"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin", keys=[4,9])
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [0,1,0,0,0,1,1,0,1,1,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 15
    colnms = ["bin3","BIN2v2","BIN2v2"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin", keys={4:1,9:0})
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1,0,1,1,1,0,0,1,0,0,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 16
    colnms = ["bin4","BIN4","BIN4"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin", keys=["jose","garcia"])
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [1,0,1,1,0,0,1,0,1,0,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 17
    colnms = ["bin4","BIN4","BIN4"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin", keys=dict(jose=0, garcia=1))
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [0,1,0,0,1,1,0,1,0,1,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 18
    colnms = ["bin5","BIN5","BIN5"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin")
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [0,1,0,np.nan,1,1,0,1,0,1,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 19
    colnms = ["bin5","BIN5","BIN5"]
    dset = DatasetManagerV3(debug_df)
    new_name = dset.validateColumn(colnms[1], mode="bin", keys={0:0, 1:1, 3:1})
    if new_name != colnms[-1]:
        raise Exception("Error at validateColumn, step={} (unexpected name)".format(step))
    expected_results = [0,1,0,1,1,1,0,1,0,1,]
    if compareCols(dset._proc_data[new_name], expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    ##### CATEGORICAL
    
    step = 20
    colnms = ["cat1","CAT1",["CAT1_0","CAT1_1","CAT1_2","CAT1_3"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat")
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = debug_df[colnms[0]]
    if compareCols(np.argmax(np.array(dset._proc_data[new_names]), axis=1), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 21
    colnms = ["cat2","CAT2",["CAT2_0","CAT2_1","CAT2_2","CAT2_3", "CAT2_8"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat")
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [0,1,3,2,1,4,2,3,2,0]
    if compareCols(argmaxna(np.array(dset._proc_data[new_names])), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
        
    step = 22
    colnms = ["cat2","CAT2",["CAT2_0","CAT2_1","CAT2_2","CAT2_3"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys=[0,1,2,3])
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [0,1,3,2,1,np.nan,2,3,2,0]
    if compareCols(argmaxna(np.array(dset._proc_data[new_names])), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 23
    colnms = ["cat2","CAT2",["CAT2_0","CAT2_1","CAT2_2","CAT2_3"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys={0:0,1:1,2:2,3:3,8:1})
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [0,1,3,2,1,1,2,3,2,0]
    if compareCols(argmaxna(np.array(dset._proc_data[new_names])), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 24
    colnms = ["cat2","CAT2",["CAT2_1","CAT2_2"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys={0:0,1:1,2:2,3:3,8:1}, min_key_freq=.25)
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [np.nan,0,np.nan,1,0,0,1,np.nan,1,np.nan]
    if compareCols(argmaxna(np.array(dset._proc_data[new_names])), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))

    step = 25
    colnms = ["cat2","CAT2",["CAT2_1","CAT2_2"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys={0:0,1:1,2:2,3:3,8:1}, min_key_count=3)
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [np.nan,0,np.nan,1,0,0,1,np.nan,1,np.nan]
    if compareCols(argmaxna(np.array(dset._proc_data[new_names])), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
    
    step = 26
    colnms = ["cat3","CAT3",["CAT3_1","CAT3_2","CAT3_3","CAT3_4","CAT3_8"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat")
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [3,0,2,1,0,4,1,2,1,3]
    if compareCols(np.argmax(np.array(dset._proc_data[new_names]), axis=1), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
    
    step = 27
    colnms = ["cat4","CAT4",["CAT4_hi","CAT4_lo","CAT4_mid"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat")
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [dict(lo=1,mid=2,hi=0)[c] for c in ['lo','mid','hi','mid','mid','lo','hi','hi','lo','mid']]
    if compareCols(np.argmax(np.array(dset._proc_data[new_names]), axis=1), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
    
    step = 28
    colnms = ["cat4","CAT4",["CAT4_0","CAT4_1","CAT4_2"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys=dict(lo=0,mid=1,hi=2))
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [dict(lo=0,mid=1,hi=2)[c] for c in ['lo','mid','hi','mid','mid','lo','hi','hi','lo','mid']]
    if compareCols(np.argmax(np.array(dset._proc_data[new_names]), axis=1), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
    
    step = 29
    colnms = ["cat4","CAT4",["CAT4_1","CAT4_2","CAT4_3"]]
    dset = DatasetManagerV3(debug_df)
    new_names = dset.validateColumn(colnms[1], mode="cat", keys=dict(lo=1,mid=2,hi=3))
    if compareCols(new_names, colnms[-1]) != True:
        raise Exception("Error at validateColumn, step={} (unexpected names)".format(step))
    expected_results = [dict(lo=0,mid=1,hi=2)[c] for c in ['lo','mid','hi','mid','mid','lo','hi','hi','lo','mid']]
    if compareCols(np.argmax(np.array(dset._proc_data[new_names]), axis=1), expected_results) != True:
        raise Exception("Error at validateColumn, step={}".format(step))
    else:
        print("Step {}: ok".format(step))
    
    ##### IMPUTATION
    
    dset = DatasetManagerV3(debug_df)
    [dset.validateColumn(c, mode="qty") for c in dset._proc_data.columns.tolist() if c[:3]=="QTY" or c[:3]=="QTE"]
    [dset.validateColumn(c, mode="bin") for c in dset._proc_data.columns.tolist() if c[:3]=="BIN"]
    [dset.validateColumn(c, mode="cat") for c in dset._proc_data.columns.tolist() if c[:3]=="CAT"]
    
    dset.ImputeMissingValues()
    
    #debug_df
    #dset._proc_data
    #dset._proc_data.columns
    
    dset = DatasetManagerV3(debug_df)
    [dset.validateColumn(c, mode="qty") for c in dset._proc_data.columns.tolist() if c[:3]=="QTY" or c[:3]=="QTE"]
    [dset.validateColumn(c, mode="bin") for c in dset._proc_data.columns.tolist() if c[:3]=="BIN"]
    [dset.validateColumn(c, mode="cat") for c in dset._proc_data.columns.tolist() if c[:3]=="CAT"]
    
    dset.ImputeMissingValues(columns=["QTY1","QTY8","BIN2v1"])
    
    np.sum(pd.isna(dset._proc_data))

    dset.ImputeMissingValues(columns=["QTY1","QTY2","CAT1_0"])
    dset.ImputeMissingValues(columns=["QTY1","QTY2","CAT1_0","CAT1_1","CAT1_2","CAT1_3"])
    
    ##### FREEZE
    
    dset = DatasetManagerV3(debug_df)
    [dset.validateColumn(c, mode="qty") for c in dset._proc_data.columns.tolist() if c[:3]=="QTY" or c[:3]=="QTE"]
    [dset.validateColumn(c, mode="bin") for c in dset._proc_data.columns.tolist() if c[:3]=="BIN"]
    [dset.validateColumn(c, mode="cat") for c in dset._proc_data.columns.tolist() if c[:3]=="CAT"]
    dset.validateColumn("Y", mode="qty", dataset="y")
    dset.ImputeMissingValues()

    dset.freezeDataset()
    
    dset.save(r"C:\Users\admin\Documents\Common\temp", "test_v3_1")
    
    dset = DatasetManagerV3(path_or_df = os.path.join(r"C:\Users\admin\Documents\Common\temp", "test_v3_1"))
    
    dset._proc_data
    
    
    
    
    
    
    
    