import pathlib
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import yaml


import rest_api.schemas
import features.util_feature
import features.feature_engineer

import rest_api.schemas

# import schemas

## paths
# BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
BASE_DIR = pathlib.Path( 'models/trained' ) ## ! if this path does work in deployment, change it according to 
## Writign Dockerfile to Pck model with FASTAPI Wrapper 2 mins
MODEL_PATH = BASE_DIR / 'greenhouse_emission_predict_model.pkl'
PREPROCESSOR_PATH = BASE_DIR / 'preprocessor.pkl'


model_pkl = joblib.load(MODEL_PATH)
preprocessor_pkl = joblib.load(PREPROCESSOR_PATH)

def predict( request: rest_api.schemas.Emission_Prediction_Request ) -> rest_api.schemas.Emission_Prediction_Response :
    """
    
    
    :param request: 
    :type request: rest_api.schemas.Emission_Prediction_Request
    :return: 
    :rtype: Emission_Prediction_Response
    """
    ## Prepare input data
    featured_df = pd.DataFrame(  [ request.model_dump() ] )

    engineered_x_df, engineered_cols = features.util_feature.transform_to_engineered_df(
        preprocessor= preprocessor_pkl, xx= featured_df,
        remaining_features_ls= features.feature_engineer.REMAINING_Features_ls 
    )

    yhat = int(  model_pkl.predict( engineered_x_df )[0]  )

    confidence_interval = [ round( yhat * 0.9, 0 ),  round( yhat * 1.1, 0) ]

    _response = rest_api.schemas.Emission_Prediction_Response(
        predicted_emission= yhat,
        confidence_interval= confidence_interval,
        feature_importance= {},
        prediction_time= datetime.now().isoformat()
    )

    return _response


def batch_predict( request_ls: list[ rest_api.schemas.Emission_Prediction_Request ] ) -> list[ int ]:
    """
    
    
    :param request_ls: 
    :type request_ls: list[ rest_api.schemas.Emission_Prediction_Request ]
    :return: 
    :rtype: list[ int ]
    """
    input_data = pd.DataFrame( [ e_req.model_dump() for e_req in request_ls ] )

    engineered_x_df, _ = features.util_feature.transform_to_engineered_df(
        preprocessor= preprocessor_pkl, xx= input_data,
        remaining_features_ls= features.feature_engineer.REMAINING_Features_ls 
    )

    yhat_ls = model_pkl.predict( engineered_x_df.values )

    return [  int(yhat) for yhat in yhat_ls  ]
