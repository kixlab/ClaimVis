import uvicorn
from fastapi import Depends, FastAPI
from models import *
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from main import Pipeline
from AutomatedViz import AutomatedViz

import log_crud, models, ORMModels
from database import SessionLocal, engine
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# df = pd.read_csv("../Datasets/owid-energy-data.csv")

"""
    Claim map has the following structure:

    claim_map = {
        "sentence_1": [
            { # each of the justification corresponds to a dataset
                "suggestions": [
                    {
                        "query": ...,
                        "visualization": ...,
                        "reasoning_steps": [...],
                        "justification": ...,
                        "value_map": {
                            "col_1": {...}, # value is a set
                            "col_2": {...},
                            ...
                        }
                    },
                    {...}
                ],
                "sub_table": pd.DataFrame(...),
                "attributes": [...]
            },
            {...}
        ],
        "sentence_2": [...],
        ...
    }
"""


"""
    Data point set has the following structure:
    [
        DataPointSet(
            statement="{value} in {date} between {country}", 
            dataPoints=[  # Two main data point of comparison
                DataPointValue(tableName="Primary energy consumption", 
                    fields = {
                        country="United States", 
                        date="2020", 
                    }
                    valueName="Nuclear energy consumption", 
                    value=1.0,
                    unit="TWh"),
                DataPointValue(tableName="Primary energy consumption", 
                    fields = {
                        country="United Kingdom", 
                        date="2020",
                    }
                    valueName="Nuclear energy consumption", 
                    value=1.0,
                    unit="TWh"),
            ],
            fields = [Field(
                name="date",
                type="temporal",
                timeUnit= "year"
            ),
            Field(
                name="country",
                type="nominal"
            )],
            ranges = Ranges(
                fields = {
                    date = { # will take the lowest and highest date from the data points
                        'date_start': {
                            'label': '2015', 
                            'value': '2015'
                        },
                        'date_end': {
                            'label': '2022',
                            'value': '2022'
                        }
                    },
                    country = ['United States', 'United Kingdom', 'Greece', 'Germany', 'South Korea', 'Japan', 'Vietnam'] # will assume that only one of these nominal attribute is present in the claim

                }
                values = [{ # will take numerical data attribute from the attribute sets
                    'label': 'Nuclear energy consumption', # human readable column name
                    'value': 'nuclear_consumption', # name in the table
                    'unit': 'TWh', # unit of measurement
                    'provenance': 'The data was from Our World in Data, which is a non-profit organization that publishes data and research on the world\'s largest problems. The data was collected from the International Energy Agency (IEA) and the United Nations (UN).' # where the data came from
                }, {
                    'label': 'Coal energy consumption',
                    'value': 'coal_consumption',
                    'unit': 'TWh',
                    'provenance': 'The data was from Our World in Data, which is a non-profit organization that publishes data and research on the world\'s largest problems. The data was collected from the International Energy Agency (IEA) and the United Nations (UN).'
                },
                {
                    'label': 'Solar energy consumption',
                    'value': 'solar_consumption',
                    'unit': 'TWh',
                    'provenance': 'The data was from Our World in Data, which is a non-profit organization that publishes data and research on the world\'s largest problems. The data was collected from the International Energy Agency (IEA) and the United Nations (UN).'
                }],
            )
        )
    ]
"""

@app.post("/potential_data_point_sets")
def potential_data_point_sets(body: UserClaimBody, verbose:bool=False, test=False) -> list[DataPointSet]:
    user_claim = body.userClaim

    if test: # for testing purposes
        attributes = ['coal_production', 'country', 'coal_prod_per_capita', 'coal_elec_per_capita', 'coal_cons_per_capita', 'year', 'coal_consumption']
        table = pd.read_csv("../Datasets/owid-energy-data.csv")
        table.columns = table.columns.str.lower()
        table = table[attributes]
        value_map = {'country': {'china'}, 'year': {'2022'}}
        new_claim = user_claim
    else:
        pipeline = Pipeline(datasrc="../Datasets")
        claim_map, claims = pipeline.run(user_claim)
        # if verbose: print(claim_map)

        reason = claim_map[claims[0]][0]
        new_claim, table, attributes, value_map, reasoning = reason["suggestions"][0]["query"], reason["sub_table"], reason["attributes"], reason["suggestions"][0]["value_map"], reason["suggestions"][0]["justification"]
        # if verbose: print(table, attributes)

    # given a table and its attributes, return the data points
    AutoViz = AutomatedViz(
                table=table, 
                attributes=attributes, 
                test=test, 
                matcher=pipeline.data_matcher if not test else None
            )
    return AutoViz.retrieve_data_points(
                        text=new_claim, 
                        value_map=value_map, 
                        reasoning=reasoning, 
                        verbose=verbose
                    )

@app.post("/get_viz_spec")
def get_viz_spec(body: GetVizSpecBody): # needs update

    skeleton_spec = {
        "width": 'container',
        "height": '450',
        "mark": {
            "type": 'bar',
            "tooltip": True,
        },
        "encoding": {
            "x": { "field": 'a', "type": 'ordinal' },
            "y": { "field": 'b', "type": 'quantitative' },
        },
        "data": { "name": 'table' },
    }

    return skeleton_spec

@app.post("/get_data")
def get_data_new(body: GetVizDataBodyNew) -> list[dict]:
    tableName = body.tableName

    df = pd.read_csv(f"../Datasets/{tableName}")

    otherFieldNames = list(map(lambda x: x, body.fields))
    dateFieldNames = [k for (k, v) in body.fields.items() if isinstance(v, DateRange)]
    dateFieldName = dateFieldNames[0] if len(dateFieldNames) > 0 else None
    ## remove date from otherFieldsNames
    otherFieldNames.remove(dateFieldName)
    if dateFieldName:
        date_start = int(body.fields[dateFieldName].date_start.value)
        date_end = int(body.fields[dateFieldName].date_end.value)
        dates = [(i) for i in range(date_start, date_end + 1)]
    else:
        dates = None
    values = list(map(lambda x: x.value, body.values))
    categories = otherFieldNames + [dateFieldName] + values # add country and year to the list of categories

    # select rows with dates
    dataframe = df[df[dateFieldName].isin(dates)] if dates is not None else df
    for of in otherFieldNames:
        otherFieldValue = list(map(lambda x: x.value, body.fields[of]))
        dataframe = dataframe[dataframe[of].isin(otherFieldValue)]
    # df.columns = df.columns.str.lower()
    dataframe = dataframe[categories]
    dataframe.rename(columns={dateFieldName: 'date'}, inplace=True)

    dataframe.fillna(0, inplace=True)

    res_dict = dataframe.to_dict(orient='records')
    for r in res_dict:
        r['fields'] = {}
        for of in otherFieldNames:
            r['fields'][of] = r[of]
            del r[of]
        if dates:
            r['fields']['date'] = r['date']
            del r['date']
        
    return res_dict

@app.post("/logs", response_model = models.Log)
def create_log(body: LogCreate, db: Session = Depends(get_db)):
    return log_crud.create_log(db=db, log=body)

@app.get("/logs", response_model = list[models.Log])
def get_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), username: str = None):
    if username:
        return log_crud.get_logs_by_user(db=db, user=username, skip=skip, limit=limit)
    else:
        return log_crud.get_logs(db=db, skip=skip, limit=limit)

@app.get('/dataset_explanation') 
def get_dataset_explanation(dataset: str, column_name: str):
    df = pd.read_csv(f"../Datasets/info/{dataset}")
    df = df[df['title'] == column_name]
    df.fillna(0, inplace=True)

    return df.to_dict(orient='records')

def main():
    # uvicorn.run(app, host="0.0.0.0", port=9889)
    claim = UserClaimBody(userClaim="For example, 50 years ago Mexico had a birthrate of almost 7, today it is 1.9.")
    l = potential_data_point_sets(claim, verbose=True, test=False)
    print(l)

if __name__ == "__main__":
    pass