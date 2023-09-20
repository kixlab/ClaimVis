# from pydantic import BaseModel, create_model
import datetime
from enum import Enum
from typing import Dict, Optional, Union, Any
from pydantic import BaseModel as PydanticBaseModel

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class DataTableEnum(str, Enum):
    fixee = "fixed"
    variable = "variable"

class OptionProps(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None
    provenance: Optional[str] = None

class Field(BaseModel):
    name: str
    type: str
    timeUnit: Optional[str] = None

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
    fields: Dict[str, Any] # Date is now moved to here

class DataPointValue(DataPoint):
    value: float
    unit: Optional[str] = None

class DataPointSet(BaseModel):
    statement: str
    tableName: str
    dataPoints: list[DataPoint]
    fields: list[Field]
    ranges: Ranges
    reasoning: Optional[str] = None

class ClaimMap(BaseModel):        
    class SuggestValue(BaseModel):
        field: str
        values: list[str]
        explain: str
        rank: int
        caution: list[str] = []

    country: list[str]
    value: list[str]
    date: list[str]
    vis: str 
    cloze_vis: str
    rephrase: str 
    suggestion: list[SuggestValue]
    mapping: Dict[str, Any]

class Dataset(BaseModel):
    name: str
    description: str
    score: float
    fields: list[str]

class UserClaimBody(BaseModel):
    userClaim: str
    paragraph: Optional[str] = None
    context: Optional[str] = 'South Korea'

class GetVizSpecBody(BaseModel):
    userClaim: str
    tableName: str
    dataPoints: list[DataPoint]

class GetVizDataBodyNew(BaseModel):
    tableName: str
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]

class GetVizDataBodyMulti(BaseModel):
    datasets: list[Dataset]
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]

class LogBase(BaseModel):
    event: str
    payload: Optional[str] = None
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

