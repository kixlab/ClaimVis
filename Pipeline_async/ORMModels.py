from sqlalchemy.orm import mapped_column, Mapped
import datetime
from typing import Annotated

from sqlalchemy import Column, Integer, String, func
from database import Base

timestamp = Annotated[
    datetime.datetime,
    mapped_column(nullable = False, server_default=func.current_timestamp())
]
class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    event = Column(String)
    payload = Column(String)
    environment = Column(String)
    client_timestamp = Column(String)
    created_at: Mapped[timestamp]
    url = Column(String)
    username = Column(String)
    