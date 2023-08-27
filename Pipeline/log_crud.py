from sqlalchemy.orm import Session

import models, ORMModels

def get_log(db: Session, log_id: int):
    return db.query(ORMModels.Log).filter(ORMModels.Log.id == log_id).first()

def get_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(ORMModels.Log).offset(skip).limit(limit).all()

def get_logs_by_user(db: Session, user: str, skip: int = 0, limit: int = 100):
    return db.query(ORMModels.Log).filter(ORMModels.Log.username == user).offset(skip).limit(limit).all()

def create_log(db: Session, log: models.LogCreate):
    db_log = ORMModels.Log(**log.model_dump())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log