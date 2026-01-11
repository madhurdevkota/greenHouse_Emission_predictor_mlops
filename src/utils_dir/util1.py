# Simple import helper that executes each non-empty import line
from typing import Iterable, Dict, List, Tuple, Union, Any
import os
from copy import deepcopy
import glob
import pathlib
import pandas as pd
import numpy as np

import sklearn


def enforce_dtypes(df, dtype_map):
    """
    Apply dtype conversions from dtype_map to df.
    dtype_map values supported: 'datetime', 'float', 'int', 'category'.
    """
    target_df = df.copy()
    for col, kind in dtype_map.items():
        if col not in target_df.columns:
            continue
        try:
            if kind == 'datetime':
                target_df[col] = pd.to_datetime(target_df[col], errors='coerce')
            elif kind == 'float':
                target_df[col] = pd.to_numeric(target_df[col], errors='coerce').astype(float)
            elif kind == 'int':
                # use pandas nullable integer to preserve NaNs
                nums = pd.to_numeric(target_df[col], errors='coerce')
                try:
                    target_df[col] = nums.astype('Int64')
                except Exception:
                    target_df[col] = nums  # fallback to numeric if Int64 fails
            elif kind == 'category':
                target_df[col] = target_df[col].astype('category')
            else:
                # unknown kind -> attempt a safe numeric coercion, else leave as-is
                try:
                    target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
                except Exception:
                    pass
        except Exception as e:
            print(f"Error converting column {col} to {kind}: {e}")

    return target_df



def impute_by_dtype( df, impute_map: Dict[str, str] ):
    """Impute missing values in `df` according to `impute_map`.

    - numeric types ('float','int','numeric') -> median
    - categorical/object types ('category','object') -> mode 
    - other kinds: left unchanged

    Args:
      df: pandas DataFrame
      impute_map: dict mapping column name -> dtype string (as used earlier)

    Returns:
      DataFrame with imputed values.
    """

    target_df = df.copy()
    summary = []

    for col, kind in impute_map.items():
        if col not in target_df.columns:
            summary.append((col, 'missing_column', None, None))
            continue
        before_missing = int(target_df[col].isna().sum())

        if kind in ('float', 'int', 'numeric'):
            try:
                median = target_df[col].median()
                target_df[col] = target_df[col].fillna(median)
                summary.append((col, 'numeric_median', before_missing, int(target_df[col].isna().sum())))
            except Exception as e:
                summary.append((col, 'error', str(e), None))
        elif kind in ('category', 'object'):
            try:
                mode_vals = target_df[col].mode(dropna=True)
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    target_df[col] = target_df[col].fillna(mode_val)
                    summary.append((col, 'categorical_mode', before_missing, int(target_df[col].isna().sum())))
                else:
                    summary.append((col, 'no_mode_found', before_missing, int(target_df[col].isna().sum())))
            except Exception as e:
                summary.append((col, 'error', str(e), None))
        else:
            # leave as is
            summary.append((col, 'left_unchanged', before_missing, int(target_df[col].isna().sum())))

    # print concise summary
    print('Imputation summary (col, action, missing_before, missing_after):')
    for row in summary:
        print(' ', row)

    return target_df



## One hot encoding udf

def OHE_func( df, categorical_col ):
    '''
    function that returns one=hot-encoded dataframe for the given categorical column list. Also removes the original column
    input:
    df = dataframe
    categorical_col = list of string of categorical columns to be one-hot encoded '''

    if isinstance( categorical_col, str ):  categorical_col = [categorical_col]
    OHEncoder = sklearn.preprocessing.OneHotEncoder( sparse_output= False, drop= 'first', dtype= np.int8 )
    OHEncoded_np = OHEncoder.fit_transform( df[categorical_col] )
    OHEncoded_df = pd.DataFrame( OHEncoded_np, index= df.index, columns= OHEncoder.get_feature_names_out(input_features=categorical_col) )

    df_out = df.drop( columns= categorical_col )\
                .merge( OHEncoded_df, left_index= True, right_index= True )
    
    return df_out