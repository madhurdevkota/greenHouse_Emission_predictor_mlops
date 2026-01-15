import json
import logging
from copy import deepcopy
import pathlib

import sklearn as skl

import numpy as np
import pandas as pd

## set up logging
logging.basicConfig(  
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger( 'create features' )


# -----------------------------
# Domain lists
# -----------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent  ## .../app/features
_DOMAIN_DIR =  _THIS_DIR.parent / 'domain'      ## # .../app/domain

_USSTATES_JSON_path = _DOMAIN_DIR / 'usstates.json'
_SOURCE_TYPE_JSON_path = _DOMAIN_DIR / 'source_types.json'

# print(_USSTATES_JSON_path)
# print(_USSTATES_JSON_path.exists())


## --------------- Start:  helper functions  ---------------

def _load_domain_values( pathlib_path:pathlib.Path, key: str) -> set[str]:
    """
    Load a set of string values from a JSON file.

    Expected JSON structure:
      { "<key>": [ ... ] }

    Notes:
    - `pathlib_path` must be a (Pathlib) path to a JSON file
    """
    if isinstance( pathlib_path, str ):  pathlib_path = pathlib.Path( pathlib_path )
    obj = json.loads( pathlib_path.read_text(encoding='utf-8') )
    return { str(x) for x in obj[key] }


def _canon_state( s: pd.Series ) -> pd.Series:
    # Canonical state used for downstream categorical encoding:
    if isinstance( s, pd.Series ):
        return s.str.strip().str.replace( ' ', '', regex= False )
    if isinstance( s, str ):
        return s.strip().replace( ' ', '' )


def _canon_source_type( s: pd.Series ) -> pd.Series:
    # Canonical source_type used for downstream categorical encoding:
    if isinstance( s, pd.Series ):
        return s.str.strip()
    if isinstance( s, str ):
        return s.strip()



def _safe_div( num: pd.Series, den: pd.Series ) -> pd.Series:
    den = den.replace( 0, np.nan )
    return num / den


def build_universal_categories( usstates_json: str, source_types_json: str ) -> tuple[list[str], list[str], list[str]]:
    """
    Build deterministic, universal category lists for:
      - state
      - source_type
      - inter = state_source_type  (cartesian product)

    Output lists are sorted and therefore stable.
    """
    states_raw = _load_domain_values( usstates_json, key= 'states' )
    sources_raw = _load_domain_values( source_types_json, key= 'source_types' )

    states = sorted( { _canon_state(s) for s in states_raw } )
    sources = sorted( { _canon_source_type(t) for t in sources_raw } )

    # Deterministic cartesian product: state-major then source-major
    inter = [ f'{st}_{so}' for st in states for so in sources ]

    return states, sources, inter




## --------------- End:  helper functions  ---------------


def create_preprocessor_fixed( usstates_json: str, source_types_json: str ):
    """
    Create a preprocessing pipeline with a FIXED OHE feature space.
    This ensures that even if the training dataset is a subset of the universal categories,
    the resulting transformed matrix has the same columns in the same order.
    """
    logger.info( 'Creating preprocessor pipeline (fixed category space)' )

    cat_col_ls = [ 'state', 'source_type', 'inter' ]
    states_cat, sources_cat, inter_cat = build_universal_categories( usstates_json, source_types_json )

    categorical_transformer = skl.pipeline.Pipeline(
        steps= [
            (
                'onehot',
                skl.preprocessing.OneHotEncoder(
                    categories= [ states_cat, sources_cat, inter_cat ],
                    handle_unknown= 'ignore',
                    sparse_output= True
                )
            ),
        ]
    )

    preprocessor = skl.compose.ColumnTransformer(
        transformers= [
            ( 'cat', categorical_transformer, cat_col_ls ),
        ],
        remainder= 'drop'
    )

    logger.info( 'Preprocessor pipeline created' )
    return preprocessor


def transform_to_engineered_df( preprocessor, xx: pd.DataFrame,
                                remaining_features_ls: list[str] ) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform xx into a single engineered feature DataFrame with stable ordering:
      [OHE columns (named)] + [remaining numeric features (named)]

    Returns:
      engineered_x_df, engineered_cols
    """
    # transform categoricals (preprocessor must already be fitted)
    x_cat = preprocessor.transform( xx )
    x_cat_arr = x_cat.toarray() if hasattr( x_cat, 'toarray' ) else x_cat

    # get one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_names = ohe.get_feature_names_out( preprocessor.transformers_[0][2] ).tolist()

    x_cat_df = pd.DataFrame( x_cat_arr, columns= ohe_names, index= xx.index )

    # remaining numeric features in fixed order
    x_rem_df = xx.filter( remaining_features_ls ).copy()

    engineered_x_df = pd.concat( [ x_cat_df, x_rem_df ], axis= 1 )
    engineered_cols = ohe_names + remaining_features_ls

    return engineered_x_df, engineered_cols



## --------------- Start:  Main utility functions  ---------------


def create_features( df: pd.DataFrame ) -> pd.DataFrame:
    """
    Create new features from existing data.

    - do NOT one-hot encode here. We do this using sklearn pipeline preprocessor so that testing will use same preprocessor.
    - drop emissions_factor to avoid leakage.
    """
    logger.info( "Creating new features" )

    if not isinstance( df, pd.DataFrame ):
        raise TypeError( f"Expected df to be a pandas DataFrame, got {type(df)}" )

    # Required base columns for feature engineering
    required_cols = set(  [
        'capacity', 'capacity_factor', 'activity',
        'area', 'pop2020',
        'state', 'source_type'  ]
    )
    if missing := sorted( required_cols - set(df.columns) ):
        raise ValueError( f'Missing required columns for feature engineering: {missing}' )

    # states_raw_set, source_types_raw_set = _load_domain_values()

    states_raw_set = _load_domain_values( _USSTATES_JSON_path, key= 'states' )
    source_types_raw_set = _load_domain_values( _SOURCE_TYPE_JSON_path, key= 'source_types' )


    # Copy once (deep) to avoid side effects
    df_copy = deepcopy( df )

    # Canonicalize categoricals first (this is what preprocessor must expect)
    df_copy['state'] = _canon_state( df_copy['state'] )
    df_copy['source_type'] = _canon_source_type( df_copy['source_type'] )

    # Domain checks AFTER canonicalization:  validate raw domain lists by canonicalizing them the same way.
    if states_raw_set:
        states_canon_set = { str(s).strip().replace(' ', '') for s in states_raw_set }
        bad_states = sorted( set(df_copy['state'].dropna().unique()) - states_canon_set )
        if bad_states:
            logger.warning(
                "Found state values not present in src/domain/usstates.json (after canonicalization): %s",
                bad_states[:25]
            )

    if source_types_raw_set:
        source_canon_set = { str(s).strip().replace('_', '') for s in source_types_raw_set }
        bad_sources = sorted( set(df_copy['source_type'].dropna().unique()) - source_canon_set )
        if bad_sources:
            logger.warning(
                "Found source_type values not present in src/domain/source_types.json (after canonicalization): %s",
                bad_sources[:25]
            )

    feature_df = (
        deepcopy( df_copy )
        # 1) transforms + ratios
        .assign(
            log1p_activity=         lambda _df: np.log1p( _df['activity'] ),
            log1p_capacity=         lambda _df: np.log1p( _df['capacity'] ),
            log1p_pop2020=          lambda _df: np.log1p( _df['pop2020'] ),
            log1p_area=             lambda _df: np.log1p( _df['area'] ),

            log1Pop_density=        lambda _df: np.log1p( _safe_div( _df['pop2020'], _df['area'] ) ),

            activity_per_capita=    lambda _df: _safe_div( _df['activity'], _df['pop2020'] ),
            activity_per_area=      lambda _df: _safe_div( _df['activity'], _df['area'] ),
            capacity_per_capita=    lambda _df: _safe_div( _df['capacity'], _df['pop2020'] ),
            capacity_density=       lambda _df: _safe_div( _df['capacity'], _df['area'] ),

            # 2) power-system structure features
            potential_output=       lambda _df: _df['capacity'] * _df['capacity_factor'],
            utilization_ratio=      lambda _df: _safe_div(
                                        _df['activity'],
                                        ( _df['capacity'] * _df['capacity_factor'] )
                                    ),
            activity_capacityFactor=lambda _df: _df['activity'] * _df['capacity_factor'],
            activity_per_capacity=  lambda _df: _safe_div( _df['activity'], _df['capacity'] ),
            activity_capacity=      lambda _df: _df['activity'] * _df['log1p_capacity'],
            capacity_factor_capacity=lambda _df: _df['capacity_factor'] * _df['log1p_capacity'],

            # 3) interaction features
            # state and source_type already canonicalized above
            inter=                  lambda _df: _df['state'].astype('string') + '_' + _df['source_type'].astype('string'),
        )
        # ## One-hot encoding for state & interaction-field - do this using sklearn pipeline preprocessor so tha testing wull use same preprocessor
        # .pipe( api.utils.OHE_func, categorical_col= ['state', 'inter'] )
        .drop( columns= ['emissions_factor'], errors= 'ignore' )  ## as using this field would leakage the data
    )

    logger.info( "Created 1.transforms + ratios  2.power-system structure  3. Interaction features." )

    return feature_df


if __name__ == '__main__':
    pass
