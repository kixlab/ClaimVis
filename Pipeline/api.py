import uvicorn
from fastapi import FastAPI
from models import *
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from main import Pipeline
from AutomatedViz import AutomatedViz

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("../Datasets/owid-energy-data.csv")

"""
    Claim map has the following structure:

    claim_map = {
        "sentence_i": [
            { # each of the justification corresponds to a dataset
                suggestions: [
                    {
                        "query": ...,
                        "visualization": ...,
                        "reasoning_steps": [...],
                        "justification": ...
                    },
                    {...}
                ],
                "sub_table": pd.DataFrame(...),
                "attributes": [...]
            },
            {...}
        ],
        "sentence_j": [...],
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
                country = ['United States', 'United Kingdom', 'Greece', 'Germany', 'South Korea', 'Japan', 'Vietnam'] # will assume that only one of these nominal attribute is present in the claim
            )
        )
    ]
"""

@app.post("/potential_data_point_sets")
def potential_data_point_sets(body: UserClaimBody, verbose:bool=False, test=True) -> list[DataPointSet]:
    user_claim = body.userClaim.lower()

    if test: # for testing purposes
        attributes = ['labor force participation rate, female (% of female population ages 15-64) (modeled ilo estimate)', 'vulnerable employment, female (% of female employment) (modeled ilo estimate)', 'coverage of unemployment benefits and almp in 2nd quintile (% of population)', 'vulnerable employment, total (% of total employment) (modeled ilo estimate)', 'children in employment, male (% of male children ages 7-14)', 'wage and salaried workers, total (% of total employment) (modeled ilo estimate)', 'children in employment, study and work (% of children in employment, ages 7-14)', 'ratio of female to male labor force participation rate (%) (national estimate)', 'unemployment with intermediate education, female (% of female labor force with intermediate education)', 'labor force with basic education (% of total working-age population with basic education)', 'labor force participation rate for ages 15-24, male (%) (national estimate)', 'unemployment, youth female (% of female labor force ages 15-24) (national estimate)', 'labor force participation rate, male (% of male population ages 15+) (modeled ilo estimate)', 'benefit incidence of social safety net programs to poorest quintile (% of total safety net benefits)', 'share of youth not in education, employment or training, total (% of youth population)', 'country_name', 'labor force with basic education, male (% of male working-age population with basic education)', 'average working hours of children, study and work, male, ages 7-14 (hours per week)', 'labor force with intermediate education, male (% of male working-age population with intermediate education)', 'labor force participation rate, male (% of male population ages 15-64) (modeled ilo estimate)', 'self-employed, female (% of female employment) (modeled ilo estimate)', 'benefit incidence of unemployment benefits and almp to poorest quintile (% of total u/almp benefits)', 'unemployment with basic education, male (% of male labor force with basic education)', 'employment in services (% of total employment) (modeled ilo estimate)', 'self-employed, male (% of male employment) (modeled ilo estimate)', 'coverage of social protection and labor programs (% of population)', 'labor force with intermediate education, female (% of female working-age population with intermediate education)', 'child employment in agriculture, female (% of female economically active children ages 7-14)', 'employers, female (% of female employment) (modeled ilo estimate)', 'labor force participation rate, total (% of total population ages 15+) (modeled ilo estimate)', 'part time employment, male (% of total male employment)', 'self-employed, total (% of total employment) (modeled ilo estimate)', 'child employment in services, male (% of male economically active children ages 7-14)', 'children in employment, female (% of female children ages 7-14)', 'unemployment, male (% of male labor force) (modeled ilo estimate)', 'labor force, total', 'average working hours of children, study and work, female, ages 7-14 (hours per week)', 'children in employment, work only, male (% of male children in employment, ages 7-14)', 'coverage of social insurance programs in richest quintile (% of population)', 'average working hours of children, working only, male, ages 7-14 (hours per week)', 'employment in industry, male (% of male employment) (modeled ilo estimate)', 'unemployment with intermediate education, male (% of male labor force with intermediate education)', 'employers, total (% of total employment) (modeled ilo estimate)', 'contributing family workers, male (% of male employment) (modeled ilo estimate)', 'unemployment with advanced education (% of total labor force with advanced education)', 'labor force participation rate, total (% of total population ages 15+) (national estimate)', 'labor force participation rate for ages 15-24, total (%) (modeled ilo estimate)', 'ratio of female to male labor force participation rate (%) (modeled ilo estimate)', 'unemployment, female (% of female labor force) (modeled ilo estimate)', 'unemployment, youth male (% of male labor force ages 15-24) (modeled ilo estimate)', 'labor force with basic education, female (% of female working-age population with basic education)', 'employers, male (% of male employment) (modeled ilo estimate)', 'employment in agriculture (% of total employment) (modeled ilo estimate)', 'coverage of social insurance programs in poorest quintile (% of population)', 'coverage of social safety net programs in poorest quintile (% of population)', 'share of youth not in education, employment or training, male (% of male youth population)', 'children in employment, unpaid family workers, male (% of male children in employment, ages 7-14)', 'labor force participation rate for ages 15-24, total (%) (national estimate)', 'labor force participation rate, female (% of female population ages 15+) (national estimate)', 'benefit incidence of social protection and labor programs to poorest quintile (% of total spl benefits)', 'children in employment, total (% of children ages 7-14)', 'female share of employment in senior and middle management (%)', 'part time employment, female (% of total female employment)', 'unemployment, male (% of male labor force) (national estimate)', 'coverage of social safety net programs in 3rd quintile (% of population)', 'coverage of unemployment benefits and almp in poorest quintile (% of population)', 'coverage of unemployment benefits and almp in 3rd quintile (% of population)', 'adequacy of unemployment benefits and almp (% of total welfare of beneficiary households)', 'employment in agriculture, female (% of female employment) (modeled ilo estimate)', 'unemployment with intermediate education (% of total labor force with intermediate education)', 'coverage of social safety net programs (% of population)', 'unemployment with basic education (% of total labor force with basic education)', 'adequacy of social safety net programs (% of total welfare of beneficiary households)', 'child employment in agriculture, male (% of male economically active children ages 7-14)', 'contributing family workers, female (% of female employment) (modeled ilo estimate)', 'coverage of social insurance programs in 4th quintile (% of population)', 'average working hours of children, working only, female, ages 7-14 (hours per week)', 'unemployment, youth total (% of total labor force ages 15-24) (national estimate)', 'benefit incidence of social insurance programs to poorest quintile (% of total social insurance benefits)', 'adequacy of social protection and labor programs (% of total welfare of beneficiary households)', 'coverage of social insurance programs in 2nd quintile (% of population)', 'employment in industry (% of total employment) (modeled ilo estimate)', 'labor force with intermediate education (% of total working-age population with intermediate education)', 'children in employment, self-employed (% of children in employment, ages 7-14)', 'average working hours of children, study and work, ages 7-14 (hours per week)', 'children in employment, self-employed, female (% of female children in employment, ages 7-14)', 'children in employment, study and work, female (% of female children in employment, ages 7-14)', 'labor force, female (% of total labor force)', 'coverage of social insurance programs (% of population)', 'unemployment, total (% of total labor force) (national estimate)', 'coverage of unemployment benefits and almp (% of population)', 'unemployment, youth total (% of total labor force ages 15-24) (modeled ilo estimate)', 'child employment in agriculture (% of economically active children ages 7-14)', 'child employment in manufacturing, male (% of male economically active children ages 7-14)', 'employment in services, female (% of female employment) (modeled ilo estimate)', 'unemployment with basic education, female (% of female labor force with basic education)', 'child employment in manufacturing, female (% of female economically active children ages 7-14)', 'coverage of social safety net programs in 2nd quintile (% of population)', 'labor force, total', 'unemployment with advanced education, male (% of male labor force with advanced education)', 'children in employment, self-employed, male (% of male children in employment, ages 7-14)', 'children in employment, unpaid family workers, female (% of female children in employment, ages 7-14)', 'vulnerable employment, male (% of male employment) (modeled ilo estimate)', 'unemployment, female (% of female labor force) (national estimate)', 'children in employment, study and work, male (% of male children in employment, ages 7-14)', 'children in employment, wage workers (% of children in employment, ages 7-14)', 'share of youth not in education, employment or training, female (% of female youth population)', 'average working hours of children, working only, ages 7-14 (hours per week)', 'unemployment, total (% of total labor force) (modeled ilo estimate)', 'labor force participation rate for ages 15-24, female (%) (national estimate)', 'child employment in services (% of economically active children ages 7-14)', 'labor force with advanced education (% of total working-age population with advanced education)', 'adequacy of social insurance programs (% of total welfare of beneficiary households)', 'children in employment, work only (% of children in employment, ages 7-14)', 'child employment in manufacturing (% of economically active children ages 7-14)', 'unemployment with advanced education, female (% of female labor force with advanced education)', 'unemployment, youth male (% of male labor force ages 15-24) (national estimate)', 'coverage of social safety net programs in 4th quintile (% of population)', 'labor force participation rate, male (% of male population ages 15+) (national estimate)', 'coverage of unemployment benefits and almp in richest quintile (% of population)', 'coverage of social insurance programs in 3rd quintile (% of population)', 'coverage of social safety net programs in richest quintile (% of population)', 'employment in agriculture, male (% of male employment) (modeled ilo estimate)', 'coverage of unemployment benefits and almp in 4th quintile (% of population)', 'labor force participation rate for ages 15-24, male (%) (modeled ilo estimate)', 'labor force participation rate, total (% of total population ages 15-64) (modeled ilo estimate)', 'employment in industry, female (% of female employment) (modeled ilo estimate)', 'labor force participation rate for ages 15-24, female (%) (modeled ilo estimate)', 'wage and salaried workers, female (% of female employment) (modeled ilo estimate)', 'labor force with advanced education, female (% of female working-age population with advanced education)', 'child employment in services, female (% of female economically active children ages 7-14)', 'children in employment, work only, female (% of female children in employment, ages 7-14)', 'children in employment, wage workers, female (% of female children in employment, ages 7-14)', 'labor force with advanced education, male (% of male working-age population with advanced education)', 'children in employment, wage workers, male (% of male children in employment, ages 7-14)', 'wage and salaried workers, male (% of male employment) (modeled ilo estimate)', 'labor force participation rate, female (% of female population ages 15+) (modeled ilo estimate)', 'unemployment, youth female (% of female labor force ages 15-24) (modeled ilo estimate)', 'children in employment, unpaid family workers (% of children in employment, ages 7-14)', 'contributing family workers, total (% of total employment) (modeled ilo estimate)', 'part time employment, total (% of total employment)', 'employment in services, male (% of male employment) (modeled ilo estimate)']
        table = pd.read_csv("../Datasets/Social Protection & Labor.csv").iloc[:5]
        table.columns = table.columns.str.lower()
        table = table[attributes]
    else:
        pipeline = Pipeline(datasrc="../Datasets")
        claim_map, claims = pipeline.run(user_claim)
        # if verbose: print(claim_map)

        reason = claim_map[claims[0]][0]
        table, attributes = reason["sub_table"], reason["attributes"]
        # if verbose: print(table, attributes)

    # given a table and its attributes, return the data points
    AutoViz = AutomatedViz(table=table, attributes=attributes)
    return AutoViz.retrieve_data_points(text=user_claim, verbose=verbose)

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
    # uvicorn.run(app, host="0.0.0.0", port=9889)
    claim = UserClaimBody(userClaim="The labor force participation rate of China is higher than that of the United State.")
    l = potential_data_point_sets(claim, verbose=True, test=False)
    print(l)
