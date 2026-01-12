import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn as skl

import features.create_features


## set up logging
logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger( 'feature-engineering' )


def load_domain_values( path_str: str, key: str ) -> list[str]:
    """
    Load a list of string values from a JSON file.

    Expected JSON structure:
      { "<key>": [ ... ] }

    Intentionally strict:
    - no path checks
    - no fallback keys
    - failures are loud and immediate
    """
    obj = json.loads( Path(path_str).read_text( encoding= 'utf-8' ) )
    return [ str(x) for x in obj[key] ]


def _canon_state( x: str ) -> str:
    # Must match create_features.py behavior: remove spaces
    return str(x).strip().replace( ' ', '' )


def _canon_source_type( x: str ) -> str:
    # Must match create_features.py behavior: remove underscores
    return str(x).strip().replace( '_', '' )


def build_universal_categories( usstates_json: str, source_types_json: str ) -> tuple[list[str], list[str], list[str]]:
    """
    Build deterministic, universal category lists for:
      - state
      - source_type
      - inter = state_source_type  (cartesian product)

    Output lists are sorted and therefore stable.
    """
    states_raw = load_domain_values( usstates_json, key= 'states' )
    sources_raw = load_domain_values( source_types_json, key= 'source_types' )

    states = sorted( { _canon_state(s) for s in states_raw } )
    sources = sorted( { _canon_source_type(t) for t in sources_raw } )

    # Deterministic cartesian product: state-major then source-major
    inter = [ f'{st}_{so}' for st in states for so in sources ]

    return states, sources, inter


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


def transform_to_engineered_df( preprocessor, xx: pd.DataFrame, remaining_features_ls: list[str] ) -> tuple[pd.DataFrame, list[str]]:
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


def main( input_file: str, output_file: str, preprocessor_file: str, usstates_json: str, source_types_json: str ):
    """Full feature engineering pipeline."""
    logger.info( f'Loading data from {input_file}' )
    df = pd.read_csv( input_file )

    # Create features (state/source_type/inter are created/canonicalized in create_features.py)
    df_featured = features.create_features.main( df )
    logger.info( f'Created featured dataset with shape: {df_featured.shape}' )

    xx = df_featured.drop( columns= [ 'emissions_quantity' ], errors= 'ignore' )  # Features only

    ## features remaining after one-hot encoding categorical variables
    REMAINING_Features_ls = [
        'capacity', 'capacity_factor', 'activity',
        'area', 'pop2020',
        'log1p_activity', 'log1p_capacity', 'log1p_pop2020', 'log1p_area',
        'log1Pop_density', 'activity_per_capita', 'activity_per_area',
        'capacity_per_capita', 'capacity_density', 'potential_output',
        'utilization_ratio', 'activity_capacityFactor', 'activity_per_capacity',
        'activity_capacity', 'capacity_factor_capacity',
    ]

    # Create the FIXED preprocessor
    preprocessor = create_preprocessor_fixed( usstates_json= usstates_json, source_types_json= source_types_json )

    # Fit once (even though categories are fixed, sklearn still needs a fitted transformer)
    preprocessor.fit( xx )

    engineered_x_df, engineered_cols = transform_to_engineered_df(
        preprocessor= preprocessor,
        xx= xx,
        remaining_features_ls= REMAINING_Features_ls
    )

    # Append target if present
    if 'emissions_quantity' in df_featured.columns:
        engineered_df = pd.concat( [ engineered_x_df, df_featured[ [ 'emissions_quantity' ] ] ], axis= 1 )
        engineered_cols = engineered_cols + [ 'emissions_quantity' ]
    else:
        engineered_df = engineered_x_df

    engineered_df.to_csv( output_file, index= False )
    logger.info( f'Saved fully preprocessed data to {output_file}' )

    # Save the preprocessor
    joblib.dump( preprocessor, preprocessor_file )
    logger.info( f'Saved preprocessor to {preprocessor_file}' )

    # Optional: log the resulting dimensionality
    logger.info( f'Engineered feature columns (incl target if present): {len(engineered_cols)}' )

    return engineered_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser( description= 'Feature Engineering Pipeline' )
    parser.add_argument( '--input', required= True, help= 'Path to processed CSV file' )
    parser.add_argument( '--output', required= True, help= 'Path for output CSV file (engineered features)' )
    parser.add_argument( '--output_Preprocessor', required= True, help= 'Path for saving the preprocessor' )

    # domain files (universal lists)
    parser.add_argument(
        '--usstates_json',
        required= False,
        default= str( ( Path(__file__).resolve().parent / 'domain' / 'usstates.json' ) ),
        help= 'Path to usstates.json (universal list)'
    )
    parser.add_argument(
        '--source_types_json',
        required= False,
        default= str( ( Path(__file__).resolve().parent / 'domain' / 'source_types.json' ) ),
        help= 'Path to source_types.json (universal list)'
    )

    args = parser.parse_args()

    main(
        input_file= args.input,
        output_file= args.output,
        preprocessor_file= args.output_Preprocessor,
        usstates_json= args.usstates_json,
        source_types_json= args.source_types_json
    )
