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

## paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'models' / 'trained'

MODEL_PATH = MODEL_DIR / 'greenhouse_emission_predict_model.pkl'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.pkl'
MODEL_CONFIG_PATH = BASE_DIR / 'configs' / 'model_config.yaml'


model_pkl = joblib.load(MODEL_PATH)
preprocessor_pkl = joblib.load(PREPROCESSOR_PATH)

def predict_emission( request: rest_api.schemas.Emission_Prediction_Request ) -> rest_api.schemas.Emission_Prediction_Response :
    """
    Docstring for predict_emission
    
    :param request: Description
    :type request: rest_api.schemas.Emission_Prediction_Request
    :return: Description
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
    Docstring for batch_predict
    
    :param request_ls: Description
    :type request_ls: list[ rest_api.schemas.Emission_Prediction_Request ]
    :return: Description
    :rtype: list[ int ]
    """
    input_data = pd.DataFrame( [ e_req.model_dump() for e_req in request_ls ] )

    engineered_x_df, _ = features.util_feature.transform_to_engineered_df(
        preprocessor= preprocessor_pkl, xx= input_data,
        remaining_features_ls= features.feature_engineer.REMAINING_Features_ls 
    )

    yhat_ls = model_pkl.predict( engineered_x_df.values )

    return [  int(yhat) for yhat in yhat_ls  ]
