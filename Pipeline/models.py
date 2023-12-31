# from pydantic import BaseModel, create_model
import datetime
from enum import Enum
from typing import Dict, Optional, Union
from pydantic import BaseModel as PydanticBaseModel

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class DataTableEnum(str, Enum):
    fixed: str = "fixed"
    variable: str = "variable"

class OptionProps(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None
    provenance: Optional[str] = None

class Field(BaseModel):
    name: str
    type: str
    timeUnit: str = None

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Field):
            return self.name == other.name
        return False

class DateRange(BaseModel):
    date_start: OptionProps
    date_end: OptionProps

class Ranges(BaseModel):
    values: list[OptionProps]
    fields: Dict[str, Union[list, DateRange]] ## Now date moved into the fields

class DataPoint(BaseModel):
    tableName: str
    valueName: str ## Now valueName is the name of the field
    fields: Dict[str, any] # Date is now moved to here

class DataPointValue(DataPoint):
    value: float
    unit: str = None

class DataPointSet(BaseModel):
    statement: str
    tableName: str
    dataPoints: list[DataPoint]
    fields: list[Field]
    ranges: Ranges
    reasoning: Optional[str] = None

class UserClaimBody(BaseModel):
    userClaim: str
    paragraph: str = None

class GetVizSpecBody(BaseModel):
    userClaim: str
    tableName: str
    dataPoints: list[DataPoint]

class GetVizDataBodyNew(BaseModel):
    tableName: str
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]

class ClaimMap(BaseModel):
    class Suggestions(BaseModel):
        country: list[str]
        value: list[str]
        datetime: list[str]

    country: list[str]
    value: list[str]
    date: list[str]
    vis: str 
    rephrase: str 
    suggestion: Suggestions

class Dataset(BaseModel):
    name: str
    description: str
    score: float
    fields: list[str]

class LogBase(BaseModel):
    event: str
    payload: str = None
    environment: str
    client_timestamp: str
    url: str
    username: Optional[str] = None

class LogCreate(LogBase):
    pass

class Log(LogBase):
    id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True
        orm_mode = True

