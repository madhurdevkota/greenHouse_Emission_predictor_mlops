import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import sklearn as skl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn

# -----------------------------
# Config logging
# -----------------------------
logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------

def parse_arguments():

    parser = argparse.ArgumentParser( description= 'Train and register final model from config.' )
    
    parser.add_argument( '--config', type= str, required= True, help = 'Path to model_config.yaml'  )
    parser.add_argument( '--data', type= str, required= True, help = 'Path to feature_engineered CSV dataset'  )
    parser.add_argument( '--models-dir', type= str, required= True, help = 'Directory to save trained model'  )
    parser.add_argument( '--mlflow-tracking-uri', type= str, default= None, help = 'MLflow tracking URI'  )

    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------

def get_model_instance( model_name: str, model_params: dict ):
    MODEL_MAP = {
        'RandomForest': skl.ensemble.RandomForestRegressor,
        'GradientBoosting': skl.ensemble.GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor
    }

    if model_name not in MODEL_MAP:
        raise ValueError( f"Model '{model_name}' is not available. Check model_name" )

    model_class = MODEL_MAP[ model_name ]
    return model_class( **model_params )

def main( args ):
    f = open( args.config, 'r' )
    config = yaml.safe_load(f)
    f.close()

    model_cfg = config['model']

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri( args.mlflow_tracking_uri )
        mlflow.set_experiment( config['model']['name'] )

    ## load dataset
    data_df = pd.read_csv( args.data )
    target = model_cfg['target_variable']

    xx = data_df.drop( columns=[ target ] )
    yy = data_df[ target ]

    xtrn, xtst, ytrn, ytst = train_test_split( xx, yy, test_size=0.2, random_state=42 )
    logger.info( f'Training dataset shape: {xtrn.shape}, Test dataset shape: {xtst.shape}' )

    ## get model instance
    model_obj = get_model_instance( model_cfg['best_model'], model_cfg['parameters'] )
    # Start MLflow run
    mlflow.start_run( run_name= 'final_training_prior_to_deployment' )
    logger.info( f'Started MLflow run with ID: {mlflow.active_run().info.run_id}' )

    logger.info( f'Training model: {model_cfg["name"]}' )
    model_obj.fit( xtrn, ytrn )

    yhat = model_obj.predict( xtst )

    mae = float(  skl.metrics.mean_absolute_error( ytst, yhat )  )
    r2 = float(  skl.metrics.r2_score( ytst, yhat )  )

    ## logs params and metrics
    mlflow.log_params( model_cfg['parameters'] )
    mlflow.log_metrics( { 'mae': mae, 'r2':r2 } )

    ## log and register the model
    mlflow.sklearn.log_model( sk_model= model_obj, name= 'model' )
    model_name = model_cfg['name']
    model_uri = f'runs:/{mlflow.active_run().info.run_id}/tuned_model'

    logger.info( f'Registering model {model_name} to MLflow Model Registry' )

    mlf_client = MlflowClient()

    try: mlf_client.create_registered_model(model_name)
    except Exception as e:  logger.warning( f'Model {model_name} may already be registered. Exception: {e}' )

    model_version = mlf_client.create_model_version( 
        name= model_name, source= model_uri,
        run_id= mlflow.active_run().info.run_id
    )

    logger.info( f'Model {model_name} registered with version: {model_version.version}' )

    ## transition model stage to 'Staging'
    mlf_client.transition_model_version_stage(
        name = model_name, version= model_version.version, stage= 'Staging'
    )
    logger.info( f'Model {model_name} version {model_version.version} transitioned to stage Staging' )

    ## add human-readable description
    description = f"""Model for predicting emissions.
    Algorithm: {model_name}
    Hyperparameters: {model_cfg.get('parameters')}
    Features used: All features in the dataset except the target variable
    Target variable: {target}
    Trained on dataset: {args.data}
    Model saved at: {args.models_dir}/trained/{model_name}.pkl
    Performance metrics:
    - MAE: {mae:.2f}
    - R²: {r2:.4f}
    """



    mlf_client.update_registered_model( name= model_name, description= description )

    ### add tags for better organization
    model_tag_ls = [
        (  model_name, 'algorithm', model_cfg['best_model'] ),
        (  model_name, 'hyperparameters', str(model_cfg['parameters']) ),
        (  model_name, 'features', 'All features except target variable' ),
        (  model_name, 'target_variable', target ),
        (  model_name, 'training_dataset', args.data ),
        (  model_name, 'model_path', f'{args.models_dir}/trained/{model_name}.pkl' )
    ]
    ### add tags to the registered model using model_tag_ls using lambda
            ### add tags to the registered model using model_tag_ls using lambda
    _ = list( map(  lambda t:
                    mlf_client.set_registered_model_tag(t[0], t[1], t[2]),
                    model_tag_ls ) )
    
    ### add dependency tags
    deps = {
        'python_version': platform.python_version(),
        'scikit_learn_version': skl.__version__,
        'xgboost_version': xgb.__version__,
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }

    _ = list( map(  lambda e_kv:
                mlf_client.set_registered_model_tag( model_name, e_kv[0], e_kv[1] ),
                deps.items()  )
    )

    ## save model locally
    model_save_path = f'{args.models_dir}/trained/{model_name}.pkl'
    joblib.dump( model_obj, model_save_path )
    logger.info( f'Model saved locally at: {model_save_path}' )

    logger.info(  f'Final MAE: {mae:.2f}, R²: {r2:.4f}'  )


    mlflow.end_run()

if __name__ == '__main__':
    args = parse_arguments()
    main( args )


## to run
## python .\src\models\train_models.py --config .\configs\model_config.yaml --data .\data\processed\feature_engineered.csv --models-dir models --mlflow-tracking-uri http://localhost:5555