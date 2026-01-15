import pathlib
import enum
import pydantic

from typing import List
import json

import features.util_feature




# -----------------------------
# Domain JSON paths
# -----------------------------


_THIS_DIR = Path(__file__).resolve().parent   # .../app
_DOMAIN_DIR = _THIS_DIR / 'domain'             # .../app/domain

_USSTATES_JSON_path = _DOMAIN_DIR / 'usstates.json'
_SOURCE_TYPE_JSON_path = _DOMAIN_DIR / 'source_types.json'

print( f'\n{_SOURCE_TYPE_JSON_path}\n' )

def build_str_enum(enum_name: str, values: list[str]) -> enum.EnumMeta:
    """
    Build a string Enum where:
      - enum member name == enum value
      - enum value is what FastAPI exposes in dropdowns
    """
    return enum.Enum(  enum_name,
                       { v: v for v in values },  type= str
                    )


# -----------------------------
# Build Enums from domain files
# -----------------------------
USState = build_str_enum(
    enum_name= 'USState',
    values= features.util_feature._load_domain_values(_USSTATES_JSON_path, key='states')
)

SourceType = build_str_enum(
    enum_name= 'SourceType',
    values= features.util_feature._load_domain_values(_SOURCE_TYPE_JSON_path, key='source_types')
)


# -----------------------------
# API Schemas
# -----------------------------
class Emission_Prediction_Request(pydantic.BaseModel):
    capacity: float = pydantic.Field(
        ..., ge=0, description='Capacity of the Industry - must be non negative numerical value'
    )
    capacity_factor: float = pydantic.Field(
        ..., ge=0, description='Capacity factor of the Industry - must be non negative numerical value'
    )
    activity: float = pydantic.Field(
        ..., ge=0, description='Activity of the Industry - must be non negative numerical value'
    )

    source_type: SourceType
    state: USState

    area: float = pydantic.Field(
        ..., ge=0, description='Area of the State - must be non negative numerical value'
    )
    pop2020: int = pydantic.Field(
        ..., ge=0, description='Population of the State - must be non negative integer value'
    )

# class SourceType( str, enum.Enum ):
#     gas = 'gas'
#     oil = 'oil'
#     coal = 'coal'
#     other_fossil = 'other_fossil'
#     biomass = 'biomass'
#     waste = 'waste'



class Emission_Prediction_Response(pydantic.BaseModel):
    predicted_emission: float
    confidence_interval: List[float]
    feature_importance: dict
    prediction_time: str
