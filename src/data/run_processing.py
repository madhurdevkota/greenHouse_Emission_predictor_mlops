## include all required libraries here

import os
import logging
from copy import deepcopy
import glob
import pathlib
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely as shpy


## import local lib

# import api.utils  

import utils_dir.util1


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_DIRPATH = PROJECT_ROOT / 'data' / 'raw'

GEO_DATA_DIRPATH =  PROJECT_ROOT  / 'data' / 'supporting_dataset' / 'US_State_geo'

shpfile_path = GEO_DATA_DIRPATH / 'cb_2016_us_state_5m.shp'

pop_filepath = GEO_DATA_DIRPATH / 'us_pop_by_state.csv'

OUTPUT_FILEPATH = PROJECT_ROOT  / 'data' / 'processed' / 'processed_data.csv'



def load_data(file_path):
    """load data from structured dirs"""
    dir_ls = glob.glob(  os.path.join( DATA_DIRPATH, '**/' ), recursive= True  )[1:]  ## [1:]  to avoid the very root dir
    ## get all files from sub-directories
    emission_dictn = dict()
    for edir in dir_ls:
        ## inside each sector dir
        file_ls = [  str(efile) for efile in pathlib.Path(edir).rglob( '*.csv' ) ]
        emissionFile_ls = [  efile for efile in file_ls if efile.endswith('_emissions-sources.csv')  ]
        sectornm = os.path.basename( os.path.normpath(edir) )
        sectornm = sectornm[:5] + '_df'  ## example: agric_df, fores_df

        concat_df = pd.DataFrame()
        for eEmission in emissionFile_ls:
            _dfi = pd.read_csv( eEmission )
            subSector = os.path.basename(eEmission).replace( '_emissions-sources.csv', '' )   ## clean subsector, ex. manure-management-cattle-feedlot
            _dfi['emission_subsector'] = subSector  ## a col value with subsector for later identification about the subsection source file
            concat_df = pd.concat(  [ concat_df, _dfi ], ignore_index= True )
        
        emission_dictn[sectornm] = concat_df  ## --> put all the section dfs into dict

    ### for this project, to make it manageable only working with Power dataset

    _power_df = deepcopy( emission_dictn['power_df'] )

    ## geospatial datasets
    _usa_gdf = gpd.read_file( shpfile_path )
    _uspop_df = pd.read_csv( pop_filepath )

    logger.info( f'Loading data from {file_path}' )

    return _power_df, _usa_gdf, _uspop_df


def process_data( _power_df: pd.DataFrame, _usa_gdf: gpd.GeoDataFrame, _uspop_df: pd.DataFrame ):
    ### write docstring
    """ Process raw data into cleaned and feature-engineered DataFrame.
    Args:
      _power_df: Raw power emissions DataFrame.
      _usa_gdf: GeoDataFrame of US states.
      _uspop_df: DataFrame of US state populations.
    Returns:
        Cleaned and DataFrame.
    """
    ## Data Processing steps

    power_df = (   deepcopy( _power_df )
        .drop( columns= [ 'iso3_country', 'original_inventory_sector', 'temporal_granularity', 'capacity_units', 'activity_units', 'created_date', 'source_name', 'other1', 'other1_def', 'emission_subsector', 'emissions_factor_units'  ] )
        .rename(  columns= {  'other2': 'biomass_emissions', 'other3': 'biomass_capacity', 'other4': 'biomass_generation' }  )
        .drop( columns= [ 'other2_def', 'other3_def', 'other4_def'  ] )

        ## explode based on 'source_type' col
        .assign(  source_type = lambda df: df['source_type'].str.split(',')  )
        .explode(  'source_type', ignore_index= True  )
        .assign( source_type = lambda df: df['source_type'].astype('category') )
        .reset_index( drop= True )

        .assign(
            ## convert object into datetime
            start_time = lambda df: pd.to_datetime( df['start_time'] ),
            end_time = lambda df: pd.to_datetime( df['end_time'] ),
            # clean 'source_type' column
            source_type = lambda df: df['source_type'].str.strip().str.lower(),
        )

        ## remove alaska and hawaai for simplicity
        .query( 'lon > -130' )

        .dropna( subset= ['emissions_quantity'] )
        ## get a geometry column using lat lon
        .assign(    geometry = lambda df: df.apply(  lambda _df: shpy.Point( _df['lon'], _df['lat'] ), axis= 1  )    )
    )

    ## convert pandas into geopandas so that it have goespatial info
    power_gpd = gpd.GeoDataFrame( power_df.copy(), geometry= 'geometry', crs= 'EPSG:2264' )
    ##  only CO2
    powerCo2_gpd = ( power_gpd.copy()
        .query( 'gas == "co2"' )
        .drop( columns= 'gas' )
        .assign(  emi_log = lambda df: np.log2( df['emissions_quantity'] )  )
        [  lambda df: df['emi_log'] != -np.inf  ]  ## remove -np.inf from emi_log
    )

    ### removing alask and other unrequired geographies for our case
    CLIP_BOX = shpy.geometry.box( -150, 25, -60, 50 )  ## xmin, ymin, xmax, ymax (longitude, latitude)
    state_san_ls = [  'Alaska', 'American Samoa', 'Guam', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands', 'Commonwealth of the Northern Mariana Islands' ]
    usa_gdf = (   deepcopy( _usa_gdf )
        [  lambda df:  ~df['NAME'].isin( state_san_ls )  ]
        ## remove clip the polygon outside clip_box for aesthetic purpose
        .assign(   geometry = lambda df: df['geometry'].intersection( CLIP_BOX )  )
    )

    ## get population data by states
    uspop_df = (  deepcopy( _uspop_df )
        .drop( columns= [ 'rank', 'percent_of_total']  )
        .replace(  { 'DC' : 'District of Columbia'  }  )
    )

    ## Spatial Join to get total emission per state

    powerUSA_gdf = (  deepcopy(powerCo2_gpd)
        .filter(  [ 'source_id', 'emissions_quantity', 'geometry', 'activity' ], axis= 'columns'  )
        ## spatial join to get which point lies within which state polygon
        .sjoin( usa_gdf[ ['STUSPS', 'NAME', 'geometry'] ], how= 'inner', predicate= 'within' )
        
        .groupby( 'NAME' )[['emissions_quantity']].sum().reset_index()  ## groupby for summerization of emissions_quantity
        .merge(  usa_gdf[['STUSPS', 'NAME', 'geometry']], how= 'right', on= 'NAME'  )
        .pipe(  lambda df: gpd.GeoDataFrame( df, geometry= 'geometry', crs= usa_gdf.crs )  )

        ## join with population data 
        .merge( uspop_df, how= 'left', left_on= 'NAME', right_on= 'state' )
        .drop(  columns= [ 'STUSPS', 'NAME' ]  )
        .rename(  columns= { '2020_census': 'pop2020' }  )

        ## normalizing the emission based on state area
        .assign(
            area = lambda df: df.area,
            ## normalized emission amount per area
            emission_AreaNorm = lambda df: df.apply( lambda _df: _df['emissions_quantity']/_df['area'], axis= 1 ),
            ## normalized emission amount per population ( emission per capita)
            emission_PopNorm = lambda df: df.apply( lambda _df: _df['emissions_quantity']/_df['pop2020'], axis= 1 )
        )
        
    )

    powerCo2_df = (  deepcopy( powerCo2_gpd )
        # .drop(  columns= [ 'geometry', 'geometry_ref' ]  )
        .drop(  columns= [ 'emi_log', 'source_id','lat', 'lon' ]  )
        .assign(   source_type = lambda df: df['source_type'].astype('category')  )

        ## spatial join with powerUSA_gdf to get state area and pop
        .sjoin(  powerUSA_gdf[ ['state', 'geometry', 'area', 'pop2020' ] ], how= 'inner', predicate= 'within'  )
        .drop(  columns= [ 'geometry', 'geometry_ref', 'index_right' ], errors= 'ignore' )
        .reset_index( drop= True )
    )

    ## Hard-code schemas enforcement
    DTYPE_MAP: dict = {
        'start_time': 'datetime',
        'end_time': 'datetime',
        'emissions_quantity': 'float',
        'emissions_factor': 'float',
        'capacity': 'float',
        'capacity_factor': 'float',
        'activity': 'float',
        'modified_date': 'datetime',
        'source_type': 'category',
        'biomass_emissions': 'float',
        'biomass_capacity': 'float',
        'biomass_generation': 'float',
        'state': 'category',
        'area': 'float',
        'pop2020': 'float'
    }

    powerCo2_df = utils_dir.util1.enforce_dtypes( powerCo2_df, DTYPE_MAP )

    print('\nApplied dtypes:')
    print( powerCo2_df[ [c for c in DTYPE_MAP.keys() if c in powerCo2_df.columns] ].dtypes )

    ## ------------------------- Impute missing values ------------------------- 
    miss_df = (
        pd.DataFrame( powerCo2_df.isna().sum() , columns= [ 'miss_count'] )
        .reset_index().rename( columns= { 'index': 'colnm' } )
        .query( 'miss_count > 0'  ).reset_index( drop= True  )
    )
    ## 30 percent threshold of total rows
    miss_30_threshold = powerCo2_df.shape[0]*30/100
    miss_col_less_threshold = miss_df.query( f'miss_count < {miss_30_threshold}' )
    miss_col_more_threshold = miss_df.query( f'miss_count >= {miss_30_threshold}' )
    logger.info( f'Columns with missing values less than 30% rows: {miss_col_less_threshold["colnm"].tolist()}' )
    logger.info( f'Columns with missing values more than 30% rows: {miss_col_more_threshold["colnm"].tolist()}' )

    # drop col which have more than 30% missing values
    powerCo2_df = powerCo2_df.drop( columns= miss_col_more_threshold['colnm'].tolist() )
    # impute col which have less than 30% missing values
    col2Impute = miss_col_less_threshold['colnm'].to_list()
    ## get key value pairs of DTYPE_MAP for elements in  col2Impute
    impute_dtype_map = {  k: v for k, v in DTYPE_MAP.items() if k in col2Impute  }
    powerCo2_df = utils_dir.util1.impute_by_dtype(powerCo2_df, impute_dtype_map)
    # show top rows and missing counts after imputation
    print('\nMissing counts after imputation:')
    print(  powerCo2_df.isna().sum().loc[ impute_dtype_map.keys() ]  )
    ## logging info
    for col in impute_dtype_map.keys():
        logger.info( f'Imputed missing values in column: {col} using dtype-based strategy for dtype: {impute_dtype_map[col]}' )
    
    return powerCo2_df


def save_processed_data( processed_df: pd.DataFrame, OUTPUT_FILEPATH: str ):
    """ Save processed DataFrame to specified path in CSV format.
    Args:
      processed_df: The processed pandas DataFrame to save.
      OUTPUT_FILEPATH: The file path where the DataFrame should be saved.
    """
    processed_df.to_csv( OUTPUT_FILEPATH, index= False )
    logger.info( f'Saved processed data to {OUTPUT_FILEPATH}' )


def main( data_dirpath, output_filepath ):
    _power_df, _usa_gdf, _uspop_df = load_data( data_dirpath )
    for df, name in [ (_power_df, 'power data'), (_usa_gdf, 'USA geo data'), (_uspop_df, 'US population data') ]:
        logger.info( f'Loaded {name} shape: {df.shape}' )

    processed__powerCo2_df = process_data( _power_df, _usa_gdf, _uspop_df )

    logger.info( f'Processed data shape: {processed__powerCo2_df.shape}' )

    save_processed_data( processed__powerCo2_df, output_filepath  )

    return processed__powerCo2_df

if __name__ == '__main__':
    main( data_dirpath= DATA_DIRPATH, output_filepath= OUTPUT_FILEPATH  )
