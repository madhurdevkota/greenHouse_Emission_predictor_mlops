from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware as fapi_CORSMiddleware

import docker_fastapi_app.inference
import docker_fastapi_app.schemas

## init fastapi app with metadata

### initialize FastAPI app

fapi_app = FastAPI(

    title= 'Green House Emission Prediction API',
    description= 'An API that serves ML model for predicting green house emission quantity based on power plant features.',
    version= '1.0.0',
    contact= {
        'name': 'Madhur Dev',
        'website': 'https://www.madhurdev.com',
        'linkedin': 'www.linkedin.com/in/madhurdev',
        'dashboard-portfolio':  'https://madhurdev.com/dashboards',
        'github': 'https://github.com/madhurdevkota'
    },
    license_info= {
        'name': 'Apache 2.0',
        'url': 'https://www.apache.org/licenses/LICENSE-2.0.html'
    }
)

### add cors middleware
fapi_app.add_middleware(
    fapi_CORSMiddleware,
    allow_origins= ['*'], allow_credentials= True,
    allow_methods= ['*'], allow_headers= ['*'],
)

## end points
## health check endpoint
@fapi_app.get( '/health', response_model = dict )
async def health_check_endpoint():
    return { 'status': 'ok', 'model_loaded': True }

## predict endpoint
@fapi_app.post( '/predict', response_model= docker_fastapi_app.schemas.Emission_Prediction_Response )
async def predict_endpoint( requests: docker_fastapi_app.schemas.Emission_Prediction_Request ):
    return docker_fastapi_app.inference.predict( request= requests )

## batch predict endpoint
@fapi_app.post( '/predict-batch', response_model= list )
async def batchPredict_endpoint( requests: list[ docker_fastapi_app.schemas.Emission_Prediction_Request ] ):
    return docker_fastapi_app.inference.batch_predict( requests= requests )

