## Intructions to build Container Image for this FastAPI App 

Create the Dockerfile in the root of the source code ('greenHouse_Emission_predictor'). 

Create Docker file:

  * Base Image : `python:3.11-slim`
  * To install dependencies: `pip install requirements.txt`
  * Port: `8000`
  * Launch Command : `uvicorn main:app --host 0.0.0.0 --port 8000`

Directory structure inside the container should look like this 

```
/app
  main.py
  schemas.py
  inference.py
  requirements.txt
  /models
     /trained
         greenhouse_emission_predict_model.pkl
         preprocessor.pkl
```

