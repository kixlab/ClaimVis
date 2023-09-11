import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Depends, FastAPI, HTTPException
from models import *
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from main import Pipeline
from AutomatedViz import AutomatedViz
from fastapi.responses import PlainTextResponse
import openai
import asyncio
from aiohttp import ClientSession

import log_crud, models, ORMModels
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from TableReasoning import TableReasoner
from DataMatching import DataMatcher

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
async def potential_data_point_sets(body: UserClaimBody, verbose:bool=True, test=False) -> list[DataPointSet]:
	user_claim = body.userClaim

	if test: # for testing purposes
		attributes = ['Total greenhouse gas emissions excluding land-use change and forestry (tonnes of carbon dioxide-equivalents per capita)']
		table = pd.read_csv("../Datasets/owid-co2.csv")
		# table.columns = table.columns.str.lower()
		table = table[attributes]
		value_map = {'date': {'02', '2022', '1960', '96'}}
		new_claim = user_claim
		reasoning = None
	else:
		pipeline = Pipeline(datasrc="../Datasets")
		# claim_map, claims = pipeline.run_on_text(user_claim)
		try:
			claim_map, claims = await pipeline.run_on_text(body, verbose=verbose)
			# if verbose: print(claim_map)
			if not claim_map[claims[0]]:
				raise HTTPException(status_code=404, detail="The pipeline cannot find valid statistical claim from the input. Please rephrase your claim.")

			reason = claim_map[claims[0]][0]
			new_claim, table, attributes, value_map, reasoning, viz_task = reason["suggestions"][0]["query"], reason["sub_table"], reason["attributes"], reason["suggestions"][0]["value_map"], reason["suggestions"][0]["justification"], reason["suggestions"][0]["visualization"]
			# if verbose: print(table, attributes)
		except openai.error.Timeout as e:
			#Handle timeout error, e.g. retry or log
			msg = (f"OpenAI API request timed out: {e}")
			raise HTTPException(status_code=408, detail=msg)
		except openai.error.APIError as e:
			#Handle API error, e.g. retry or log
			msg = (f"OpenAI API returned an API Error: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except openai.error.APIConnectionError as e:
			#Handle connection error, e.g. check network or log
			msg = (f"OpenAI API request failed to connect: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except openai.error.InvalidRequestError as e:
			#Handle invalid request error, e.g. validate parameters or log
			msg = (f"OpenAI API request was invalid: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except openai.error.AuthenticationError as e:
			#Handle authentication error, e.g. check credentials or log
			msg = (f"OpenAI API request was not authorized: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except openai.error.PermissionError as e:
			#Handle permission error, e.g. check scope or log
			msg = (f"OpenAI API request was not permitted: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except openai.error.RateLimitError as e:
			#Handle rate limit error, e.g. wait or log
			msg = (f"OpenAI API request exceeded rate limit: {e}")
			raise HTTPException(status_code=500, detail=msg)
		except HTTPException as e:
			raise e
		except Exception as e:
			msg = f""
			raise HTTPException(status_code=500, detail=msg)

	try:
		# given a table and its attributes, return the data points
		AutoViz = AutomatedViz(
					table=table, 
					attributes=attributes, 
					matcher=pipeline.data_matcher if not test else None
				)
		return await AutoViz.retrieve_data_points(
							text=viz_task, 
							value_map=value_map, 
							reasoning=reasoning, 
							verbose=verbose
						)
	except openai.error.Timeout as e:
		#Handle timeout error, e.g. retry or log
		msg = (f"OpenAI API request timed out: {e}")
		raise HTTPException(status_code=408, detail=msg)
	except openai.error.APIError as e:
		#Handle API error, e.g. retry or log
		msg = (f"OpenAI API returned an API Error: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except openai.error.APIConnectionError as e:
		#Handle connection error, e.g. check network or log
		msg = (f"OpenAI API request failed to connect: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except openai.error.InvalidRequestError as e:
		#Handle invalid request error, e.g. validate parameters or log
		msg = (f"OpenAI API request was invalid: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except openai.error.AuthenticationError as e:
		#Handle authentication error, e.g. check credentials or log
		msg = (f"OpenAI API request was not authorized: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except openai.error.PermissionError as e:
		#Handle permission error, e.g. check scope or log
		msg = (f"OpenAI API request was not permitted: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except openai.error.RateLimitError as e:
		#Handle rate limit error, e.g. wait or log
		msg = (f"OpenAI API request exceeded rate limit: {e}")
		raise HTTPException(status_code=500, detail=msg)
	except Exception as e:
		msg = f"Error: {e}"
		raise HTTPException(status_code=500, detail=msg)

@app.post("/potential_data_point_sets_2")
async def potential_data_point_sets_2(claim_map: ClaimMap, datasets: list[Dataset], verbose:bool=True) -> list[DataPointSet]:
	dm = DataMatcher(datasrc="../Datasets")
	tb = TableReasoner(datamatcher=dm)

	# 1. Infer the most related attributes
	dataset = datasets[0] # take the first one for now
	embed_dict = dm.attrs_embeddings[dataset.name]
	embeddings = [embed_dict[attr] for attr in dataset.fields]
	attributes = [val.rephrase for val in claim_map.value]
	scores = cosine_similarity(dm.encode(attributes), embeddings)
	argmax_indices = scores.argmax(axis=1)
	attributes = [dataset.fields[i] for i in argmax_indices]
	if verbose: print("Attributes:", attributes)

	# 2. Infer the @() countries
	table = dm.load_table(dataset.name)
	distinct_fields = dataset.fields + [p for p in ["country_name", "date"] if p not in dataset.fields]
	table = {"data": table[distinct_fields], "name": dataset.name}

	countries, infer_country_tasks = set(claim_map.country), []
	for country in countries.copy():
		if country.startswith('@('):
			if any(p in country for p in ["Bottom", "Top", "with"]):
				infer_country_tasks.append(
					tb._infer_country(
						country[2:-1], claim_map.datetime, 
						attributes, table["data"] 
					)
				)
			else: # query like @(Asian countries?) have been handled by the _suggest_variable module
				countries.update(claim_map.suggestion.country)

	inferred_countries = await asyncio.gather(*infer_country_tasks)
	[countries.update(country_list) for country_list in inferred_countries]
	claim_map.country = list(countries)
	if verbose: print("Countries:", countries)

	# 3. return DataPointSet
	av = AutomatedViz(table=table, attributes=distinct_fields, matcher=dm)
	return await av.retrieve_data_points_2(claim_map, attributes, verbose=verbose)


@app.post("/get_datasets")
async def get_relevant_datasets(claim_map: ClaimMap, verbose:bool=True):
	suggest_keywords = [keyword for sublist in claim_map.suggestion.value for keyword in sublist.values]
	keywords = suggest_keywords + [p.rephrase for p in claim_map.value]
	dm = DataMatcher(datasrc="../Datasets")
	top_k_datasets = await dm.find_top_k_datasets(claim_map.rephrase, k=5, method="gpt", verbose=verbose, keywords=keywords)
	return [Dataset(name=name, description=description, score=score, fields=fields) 
        for name, description, score, fields in top_k_datasets]

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
	if 'value' in df.columns:
		df = df[df['value'] == column_name]
		return df['Longdefinition'].iloc[0]
	elif 'title' in df.columns:
		df = df[df['title'] == column_name]
		return df['description'].iloc[0]
	else:
		return ''

@app.post('/reasoning_evaluation')
def get_reasoning_evaluation(reasoning: str):
	# activate evaluation only when users click on the reasoning dropdown or call it right after the pipeline returned the data points
	reasoner = TableReasoner()
	return reasoner._evaluate_soundness(reasoning)

@app.post('/suggest_queries')
async def get_suggested_queries(claim: UserClaimBody):
	tagged_claim = await TableReasoner()._suggest_queries_2(claim)
	return ClaimMap(**tagged_claim)

@app.get('/robots.txt', response_class=PlainTextResponse)
def robots():
	data = """User-agent: *\nDisallow: /"""
	return data

async def main():
	openai.aiosession.set(ClientSession())
	# uvicorn.run(app, host="0.0.0.0", port=9889)
	# paragraph = "Since 1960, the number of deaths of children under the age of 5 has decreased by 60%. This is thanks to the efforts of the United Nations and the World Health Organization, which have been working to improve the health of children in developing countries. They have donated 5 billion USD worth of food and clothes to Africa since 1999. As a result, African literacy increased by 20% in the last 10 years. "
	# paragraph = "South Korea’s emissions did not peak until 2018, almost a decade after Mr Lee made his commitment and much later than in most other industrialised countries. The country subsequently adopted a legally binding commitment to reduce its emissions by 40% relative to their 2018 level by 2030, and to achieve net-zero emissions by 2050. But this would be hard even with massive government intervention. To achieve its net-zero target South Korea would have to reduce emissions by an average of 5.4% a year. By comparison, the EU must reduce its emissions by an average of 2% between its baseline year and 2030, while America and Britain must achieve annual cuts of 2.8%."
	paragraph = ""
	userClaim = "Vietnam used to have a fertility rate twice the global average."
	# A significant amount of New Zealand's GDP comes from tourism
	claim = UserClaimBody(userClaim=userClaim, paragraph=paragraph)
	claim_map = await get_suggested_queries(claim)
	top_k_datasets = await get_relevant_datasets(claim_map)
	await potential_data_point_sets_2(claim_map, top_k_datasets)
	
	await openai.aiosession.get().close()

if __name__ == "__main__":
	asyncio.run(main())