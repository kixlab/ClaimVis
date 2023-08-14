import uvicorn
from fastapi import FastAPI
from models import *
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("./Datasets/owid-energy-data.csv")

@app.post("/potential_data_point_sets")
def potential_data_point_sets(body: UserClaimBody) -> list[DataPointSet]:
    user_claim = body.userClaim

    test_data = [
        DataPointSet(statement="{value} in {date} between {country}", 
                    dataPoints=[  # Two main data point of comparison
                        DataPointValue(tableName="Primary energy consumption", 
                            country="United States", 
                            date="2020", 
                            category="Nuclear energy consumption", 
                            value=1.0,
                            unit="TWh"),
                        DataPointValue(tableName="Primary energy consumption", 
                           country="United Kingdom", 
                           date="2020", 
                           category="Nuclear energy consumption", 
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
                        date = {
                            'date_start': {
                                'label': '2015', 
                                'value': '2015'
                            },
                            'date_end': {
                                'label': '2022',
                                'value': '2022'
                            }
                        },
                        values = [{
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
                        country = ['United States', 'United Kingdom', 'Greece', 'Germany', 'South Korea', 'Japan', 'Vietnam']
                    )
                )
    ]

    return test_data

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

@app.post("/get_data_old")
def get_data(body: GetVizDataBody) -> str:
    countries = body.countries
    date_start = body.date_start
    date_end = body.date_end
    dates = [(i) for i in range(date_start, date_end + 1)]
    categories = ['country', 'year'] + body.categories # add country and year to the list of categories

    # select rows with dates
    dataframe = df[df['year'].isin(dates)]
    # select rows with countries
    dataframe = dataframe[dataframe['country'].isin(countries)]
    # select columns with categories
    dataframe = dataframe[categories]
    dataframe.rename(columns={'year': 'date'}, inplace=True)


    res_json = dataframe.to_json(orient='records')
    return res_json

@app.post("/get_data")
def get_data_new(body: GetVizDataBodyNew) -> str:

    otherFieldNames = list(map(lambda x: x, body.otherFields))

    date_start = int(body.date.date_start.value)
    date_end = int(body.date.date_end.value)
    values = list(map(lambda x: x.value, body.values))
    dates = [(i) for i in range(date_start, date_end + 1)]
    categories = otherFieldNames + ['year'] + values # add country and year to the list of categories

    # select rows with dates
    dataframe = df[df['year'].isin(dates)]
    # select rows with countries
    for of in otherFieldNames:
        otherFieldValue = list(map(lambda x: x.value, body.otherFields[of]))
        dataframe = dataframe[dataframe[of].isin(otherFieldValue)]
    # dataframe = dataframe[dataframe['country'].isin(countries)]
    # select columns with categories
    dataframe = dataframe[categories]
    dataframe.rename(columns={'year': 'date'}, inplace=True)


    res_json = dataframe.to_json(orient='records')
    return res_json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9889)
    
