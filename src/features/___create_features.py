import pandas as pd
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy

## set up logging
logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger( 'create features' )

def main( df ):
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
        # ## One-hot encoding for state & interaction-field - we wil do this using sklearn pipeline preprocessor so tha testing wull use same preprocessor
        ## .pipe( api.utils.OHE_func, categorical_col= ['state', 'inter']  )
        .drop( columns= ['emissions_factor'], errors= 'ignore' ) ## as using this field would leakage the data
    )

    logger.info( "Created 1.transforms + ratios  2.power-system structure  3. Interaction features." )

    return feature_df

if __name__ == "__main__":
    # Example usage
    data = {
        'capacity': [100, 200],
        'capacity_factor': [0.3, 0.5],
        'activity': [2500, 6000],
        'pop2020': [10000, 20000],
        'area': [50, 80],
        'state': ['California', 'Texas'],
        'source_type': ['gas', 'coal'],
        'emissions_factor': [0.1, 0.2]  ## This column will be dropped
    }
    df = pd.DataFrame(data)

    featured_df = main(df)
    print(featured_df)
