from pydantic import BaseModel, create_model
from enum import Enum
from typing import Dict

class DataTableEnum(str, Enum):
    fixed: str = "fixed"
    variable: str = "variable"

class OptionProps(BaseModel):
    label: str
    value: str
    unit: str = None
    provenance: str = None

class Field(BaseModel):
    name: str
    type: str

class DateRange(BaseModel):
    date_start: OptionProps
    date_end: OptionProps

class Ranges(BaseModel):
    date: DateRange
    values: list[OptionProps]

class DataPoint(BaseModel):
    tableName: str
    country: str
    date: str
    category: str

class DataPointValue(DataPoint):
    value: float
    unit: str = None

class DataPointSet(BaseModel):
    statement: str
    dataPoints: list[DataPoint]
    fields: list[Field]
    ranges: Ranges

class UserClaimBody(BaseModel):
    userClaim: str

class GetVizSpecBody(BaseModel):
    userClaim: str
    tableName: str
    dataPoints: list[DataPoint]

class GetVizDataBody(BaseModel):
    countries: list[str]
    date_start: int
    date_end: int
    categories: list[str]
    
class GetVizDataBodyNew(BaseModel):
    date: DateRange
    values: list[OptionProps]
    otherFields: Dict[str, list[OptionProps]]