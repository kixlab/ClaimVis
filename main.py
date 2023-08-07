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
        DataPointSet(statement="Primary energy consumption", 
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
                    category = 'fixed',
                    country = 'variable',
                    date = 'fixed',
                    allCategories = ['Nuclear energy consumption', 'Coal energy consumption', 'Solar energy consumption'], ## All relelant field names in the dataset? human readable ones?
                    allCountries = ['United States', 'United Kingdom', 'Greece', 'Germany', 'South Korea', 'Japan', 'Vietnam'], # basically all countries in the dataset
                    date_start = 2000,
                    date_end = 2020 # basically all years in the dataset
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

@app.post("/get_data")
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

    res_json = dataframe.to_json(orient='records')
    return res_json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9889)
