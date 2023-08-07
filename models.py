from pydantic import BaseModel
from enum import Enum

class DataTableEnum(str, Enum):
    fixed: str = "fixed"
    variable: str = "variable"

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
    
