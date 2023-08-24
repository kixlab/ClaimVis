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
    user_claim = body.userClaim.lower()

    if test: # for testing purposes
        attributes = ['labor force participation rate, female (% of female population ages 15-64) (modeled ilo estimate)', 'vulnerable employment, female (% of female employment) (modeled ilo estimate)', 'coverage of unemployment benefits and almp in 2nd quintile (% of population)', 'vulnerable employment, total (% of total employment) (modeled ilo estimate)', 'children in employment, male (% of male children ages 7-14)', 'wage and salaried workers, total (% of total employment) (modeled ilo estimate)', 'children in employment, study and work (% of children in employment, ages 7-14)', 'ratio of female to male labor force participation rate (%) (national estimate)', 'unemployment with intermediate education, female (% of female labor force with intermediate education)', 'labor force with basic education (% of total working-age population with basic education)', 'labor force participation rate for ages 15-24, male (%) (national estimate)', 'unemployment, youth female (% of female labor force ages 15-24) (national estimate)', 'labor force participation rate, male (% of male population ages 15+) (modeled ilo estimate)', 'benefit incidence of social safety net programs to poorest quintile (% of total safety net benefits)', 'share of youth not in education, employment or training, total (% of youth population)', 'country_name', 'labor force with basic education, male (% of male working-age population with basic education)', 'average working hours of children, study and work, male, ages 7-14 (hours per week)', 'labor force with intermediate education, male (% of male working-age population with intermediate education)', 'labor force participation rate, male (% of male population ages 15-64) (modeled ilo estimate)', 'self-employed, female (% of female employment) (modeled ilo estimate)', 'benefit incidence of unemployment benefits and almp to poorest quintile (% of total u/almp benefits)', 'unemployment with basic education, male (% of male labor force with basic education)', 'employment in services (% of total employment) (modeled ilo estimate)', 'self-employed, male (% of male employment) (modeled ilo estimate)', 'coverage of social protection and labor programs (% of population)', 'labor force with intermediate education, female (% of female working-age population with intermediate education)', 'child employment in agriculture, female (% of female economically active children ages 7-14)', 'employers, female (% of female employment) (modeled ilo estimate)', 'labor force participation rate, total (% of total population ages 15+) (modeled ilo estimate)', 'part time employment, male (% of total male employment)', 'self-employed, total (% of total employment) (modeled ilo estimate)', 'child employment in services, male (% of male economically active children ages 7-14)', 'children in employment, female (% of female children ages 7-14)', 'unemployment, male (% of male labor force) (modeled ilo estimate)', 'labor force, total', 'average working hours of children, study and work, female, ages 7-14 (hours per week)', 'children in employment, work only, male (% of male children in employment, ages 7-14)', 'coverage of social insurance programs in richest quintile (% of population)', 'average working hours of children, working only, male, ages 7-14 (hours per week)', 'employment in industry, male (% of male employment) (modeled ilo estimate)', 'unemployment with intermediate education, male (% of male labor force with intermediate education)', 'employers, total (% of total employment) (modeled ilo estimate)', 'contributing family workers, male (% of male employment) (modeled ilo estimate)', 'unemployment with advanced education (% of total labor force with advanced education)']
        table = pd.read_csv("../Datasets/Social Protection & Labor.csv").iloc[:100]
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
    AutoViz = AutomatedViz(table=table, attributes=attributes, test=test)
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
@app.post("/get_data")
def get_data_new(body: GetVizDataBodyNew) -> list[dict]:

    otherFieldNames = list(map(lambda x: x, body.fields))
    ## remove date from otherFieldsNames
    otherFieldNames.remove('date')
    if 'date' in body.fields:
        date_start = int(body.fields['date'].date_start.value)
        date_end = int(body.fields['date'].date_end.value)
        dates = [(i) for i in range(date_start, date_end + 1)]
    else:
        dates = None
    values = list(map(lambda x: x.value, body.values))
    categories = otherFieldNames + ['year'] + values # add country and year to the list of categories

    # select rows with dates
    dataframe = df[df['year'].isin(dates)] if dates is not None else df
    for of in otherFieldNames:
        otherFieldValue = list(map(lambda x: x.value, body.fields[of]))
        dataframe = dataframe[dataframe[of].isin(otherFieldValue)]
    dataframe = dataframe[categories]
    dataframe.rename(columns={'year': 'date'}, inplace=True)

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


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=9889)
    claim = UserClaimBody(userClaim="Birth rates are declining worldwide.")
    l = potential_data_point_sets(claim, verbose=True, test=False)
    print(l)
