from typing import List
import pathlib
import pydantic
import enum



## get states list from a maintained txt file
def load_us_states():
    states_file = pathlib.Path('us_states.txt')
    state_ls = states_file.read_text().splitlines()
    state_dictn = {   e_state.replace(' ', '_'): e_state  for e_state in state_ls   }
    return state_dictn

USState = enum.Enum( 'USState', load_us_states(), type= str )

## source type enum
class SourceType( str, enum.Enum ):
    gas = 'gas'
    oil = 'oil'
    coal = 'coal'
    other_fossil = 'other_fossil'
    biomass = 'biomass'
    waste = 'waste'


class Emission_Prediction_Request( pydantic.BaseModel ):
     capacity: float = pydantic.Field( ..., ge= 0, description= 'Capacity of the Industry - must be non negative numerical value' )
     capacity_factor: float = pydantic.Field( ..., ge= 0, description= 'Capacity factor of the Industry - must be non negative numerical value' )
     activity: float = pydantic.Field( ..., ge= 0, description= 'Activity of the Industry - must be non negative numerical value' )
     source_type: SourceType
     state: USState
     area: float = pydantic.Field( ..., ge= 0, description= 'Area of the State - must be non negative numerical value' )
     pop: int = pydantic.Field( ..., ge= 0, description= 'Population of the State - must be non negative integer value' )


class Emission_Prediction_Response( pydantic.BaseModel ):
     predicted_emission: float 
     confidence_interval: List[ float ]
     feature_importance: dict
     prediction_time: str
