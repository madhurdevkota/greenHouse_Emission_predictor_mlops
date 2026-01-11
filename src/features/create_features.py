import json
import logging
from copy import deepcopy
from pathlib import Path
import pathlib

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
# _THIS_DIR = Path( __file__ ).resolve().parent
source_dir = pathlib.Path( 'src' )
_DOMAIN_DIR = source_dir / 'domain'

_SOURCE_TYPES_JSON = _DOMAIN_DIR / 'source_types.json'
_USSTATES_JSON = _DOMAIN_DIR / 'usstates.json'


def _read_json_list( path: Path, key_hint: str ) -> list[str]:
    """
    Read a JSON file that contains a list of strings.
    Supported shapes:
      - {"<key>": [ ... ]}
      - [ ... ]
    """
    if not path.exists():
        # Keep feature creation usable even if domain files aren't present in some contexts (tests, notebooks)
        logger.warning( f'Domain file not found: {path}. Skipping domain validation.' )
        return []

    obj = json.loads( path.read_text( encoding= 'utf-8' ) )

    if isinstance( obj, list ):
        return [ str(x) for x in obj ]

    if isinstance( obj, dict ):
        # Prefer obvious keys; else pick the first list value.
        for k in ( key_hint, 'values', 'items', 'data' ):
            if k in obj and isinstance( obj[k], list ):
                return [ str(x) for x in obj[k] ]

        for v in obj.values():
            if isinstance( v, list ):
                return [ str(x) for x in v ]

    raise ValueError( f"Unsupported JSON structure in {path}. Expected a list or dict-of-list." )


def _load_domain_values():
    """
    Loads domain values from:
      - src/domain/source_types.json
      - src/domain/usstates.json

    Returns:
      (states_raw_set, source_types_raw_set)
    """
    states = _read_json_list( _USSTATES_JSON, key_hint= 'states' )
    source_types = _read_json_list( _SOURCE_TYPES_JSON, key_hint= 'source_types' )
    return set(states), set(source_types)


def _canon_state( s: pd.Series ) -> pd.Series:
    # Canonical state used for downstream categorical encoding:
    # Remove spaces to match your existing behavior.
    return s.astype( 'string' ).str.strip().str.replace( ' ', '', regex= False )


def _canon_source_type( s: pd.Series ) -> pd.Series:
    # Canonical source_type used for downstream categorical encoding:
    # Remove underscores to match  existing behavior (e.g., other_fossil -> otherfossil).
    # return s.astype( 'string' ).str.strip().str.replace( '_', '', regex= False )
    return s.astype( 'string' ).str.strip()


def _safe_div( num: pd.Series, den: pd.Series ) -> pd.Series:
    den = den.replace( 0, np.nan )
    return num / den


def main( df: pd.DataFrame ) -> pd.DataFrame:
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
    missing = sorted( required_cols - set(df.columns) )
    if missing:
        raise ValueError( f"Missing required columns for feature engineering: {missing}" )

    # Optional domain validation (strictness kept low: warn, don't break training runs)
    states_raw_set, source_types_raw_set = _load_domain_values()

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
    ## Example usage
    data = {
        'capacity': [100, 200],
        'capacity_factor': [0.3, 0.5],
        'activity': [2500, 6000],
        'pop2020': [10000, 20000],
        'area': [50, 80],
        'state': ['California', 'New York'],
        'source_type': ['gas', 'other_fossil'],
        'emissions_factor': [0.1, 0.2]  ## This column will be dropped
    }
    df = pd.DataFrame( data )

    featured_df = main( df )
    print( featured_df.head() )
