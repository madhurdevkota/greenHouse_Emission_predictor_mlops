import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn as skl

from  features import util_feature


## set up logging
logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger( 'feature-engineering' )




def main( input_file: str, output_file: str, output_Preprocessor: str, usstates_json: str, source_types_json: str ):
    """Full feature engineering pipeline."""
    logger.info( f'Loading data from {input_file}' )
    df = pd.read_csv( input_file )

    # Create features (state/source_type/inter are created/canonicalized in create_features.py)
    df_featured = util_feature.create_features( df )
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
    preprocessor = util_feature.create_preprocessor_fixed( usstates_json= usstates_json, source_types_json= source_types_json )

    # Fit once (even though categories are fixed, sklearn still needs a fitted transformer)
    preprocessor.fit( xx )

    engineered_x_df, engineered_cols = util_feature.transform_to_engineered_df(
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
    joblib.dump( preprocessor, output_Preprocessor )
    logger.info( f'Saved preprocessor to {output_Preprocessor}' )

    # log the resulting dimensionality
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
        default= r'src/domain/usstates.json',
        help= 'Path to usstates.json (universal list)'
    )
    parser.add_argument(
        '--source_types_json',
        required= False,
        default= r'src/domain/source_types.json',
        help= 'Path to source_types.json (universal list)'
    )

    args = parser.parse_args()

    main(
        input_file= args.input,
        output_file= args.output,
        output_Preprocessor= args.output_Preprocessor,
        usstates_json= args.usstates_json,
        source_types_json= args.source_types_json
    )
