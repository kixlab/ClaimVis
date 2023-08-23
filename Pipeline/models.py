# from pydantic import BaseModel, create_model
from enum import Enum
from typing import Dict
from pydantic import BaseModel as PydanticBaseModel
from typing import Optional

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

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
    timeUnit: str = None

class DateRange(BaseModel):
    date_start: OptionProps
    date_end: OptionProps

class Ranges(BaseModel):
    date: Optional[DateRange]
    values: list[OptionProps]
    otherFields: Dict[str, list]

class DataPoint(BaseModel):
    tableName: str
    date: str
    category: str
    otherFields: Dict[str, any]

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
    paragraph: str = None

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