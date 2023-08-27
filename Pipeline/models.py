# from pydantic import BaseModel, create_model
from enum import Enum
from typing import Dict, Union
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

class GetVizDataBodyNew(BaseModel):
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]