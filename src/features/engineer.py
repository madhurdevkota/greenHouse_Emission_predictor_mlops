import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sklearn as skl
from copy import deepcopy
import joblib


## set up logging
logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger( 'feature-engineering' )



def create_features( df ):
    """Create new features from existing data."""
    logger.info( "Creating new features" )

    df_copy = deepcopy( df )

    feature_df = (  deepcopy( df_copy )
        # 1) transforms + ratios 
        .assign(
            log1p_activity=         lambda df:  np.log1p(  df['activity']  ),
            log1p_capacity=         lambda df:  np.log1p(  df['capacity']  ),
            log1p_pop2020=          lambda df:  np.log1p(  df['pop2020']  ),
            log1p_area=             lambda df:  np.log1p(  df['area']  ),
            log1Pop_density=      	lambda df:  np.log1p(  df['pop2020'] / df['area'].replace(  0, np.nan  )  ),
            activity_per_capita=    lambda df:  df['activity'] / df['pop2020'].replace(  0, np.nan  ),
            activity_per_area=      lambda df:  df['activity'] / df['area'].replace(  0, np.nan  ),
            capacity_per_capita=    lambda df:  df['capacity'] / df['pop2020'].replace(  0, np.nan  ),
            capacity_density=       lambda df:  df['capacity'] / df['area'].replace(  0, np.nan  ),

        # 2) power-system structure features 
            potential_output=               lambda df:  df['capacity'] * df['capacity_factor'],
            utilization_ratio=              lambda df:  df['activity'] / (  (df['capacity'] * df['capacity_factor']).replace(  0, np.nan  )  ),
            activity_capacityFactor=        lambda df:  df['activity'] * df['capacity_factor'],
            activity_per_capacity=          lambda df:  df['activity'] / df['capacity'].replace(  0, np.nan  ),
            activity_capacity=              lambda df:  df['activity'] * df['log1p_capacity'],
            capacity_factor_capacity=       lambda df: df['capacity_factor'] * df['log1p_capacity'],
        ## 3. interaction features
            state =  lambda df: df['state'].str.replace( ' ', '' ),
            source_type =  lambda df: df['source_type'].str.replace( '_', '' ),
            inter =  lambda df: df.apply( lambda _df: f"{_df['state']}_{_df['source_type']}", axis= 'columns'   ) ,
        
        )
        # ## One-hot encoding for state & interaction-field
        # .pipe( api.utils.OHE_func, categorical_col= ['state', 'inter']  )
        .drop( columns= ['emissions_factor'] ) ## as using this field would leakage the data
    )

    logger.info( "Created 1.transforms + ratios  2.power-system structure  3. Interaction features. Next One Hot Encoding" )

    return feature_df


def create_preprocessor():
    """Create a preprocessing pipeline."""
    logger.info( 'Creating preprocessor pipeline') 
    # Define feature groups
    cat_col_ls = ['state', 'source_type']

    # Preprocessing for categorical features
    categorical_transformer = skl.pipeline.Pipeline(
        steps= [
                    (  'onehot', skl.preprocessing.OneHotEncoder( handle_unknown='ignore' )  )
        ]
    )
    
    preprocessor = skl.compose.ColumnTransformer(
        transformers= [
                        (  'cat', categorical_transformer, cat_col_ls  )
        ]
    )
    
    logger.info( 'Preprocessor pipeline created' )

    return preprocessor


def main( input_file, output_file, preprocessor_file ):
    """Full feature engineering pipeline."""
    # Load cleaned data
    logger.info( f"Loading data from {input_file}" )
    df = pd.read_csv( input_file )

    # Create features
    df_featured = create_features( df )
    logger.info( f"Created featured dataset with shape: {df_featured.shape}" )

    # Create and fit the preprocessor
    preprocessor = create_preprocessor()
    xx = df_featured.drop( columns= ['emissions'] , errors= 'ignore' )  # Features only
    yy = df_featured['emissions']  # Target variable
    
    x_transformed = preprocessor.fit( xx )
    logger.info( "Fitted preprocessor on featured data" )

    ## Save the preprocessor
    joblib.dump( preprocessor, preprocessor_file )
    logger.info( f"Saved preprocessor to {preprocessor_file}" )



    ## save transformed xx and yy for reference/history
    df_transformed = pd.DataFrame( x_transformed )
    df_transformed['emissions'] = yy.values

    df_transformed.to_csv( output_file, index= False )
    logger.info( f'Saved fully preprocessed data to {output_file}' )

    return df_transformed  ## in actual pipeline, returning should be None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser( description= 'Feature Engineering Pipeline' )
    parser.add_argument( '--input', required= True, help= 'Path to processed CSV file' )
    parser.add_argument( '--output', required= True, help= 'Path for output CSV file (engineered features)' )
    parser.add_argument( '--output_Preprocessor', required= True, help= 'Path for saving the preprocessor' )

    args = parser.parse_args()

    main( args.input, args.output, args.output_Preprocessor )
