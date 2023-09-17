import random
import numpy as np
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
from Gloc.utils.normalizer import _get_matched_cells
from pyinstrument import Profiler
from Gloc.utils.async_llm import Model

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
	table, _, _, fields, _, info_table = dm.merge_datasets(datasets, change_dataset=True)
	new_attributes = [claim_map.mapping[attr] for attr in claim_map.value]	

	# 3. return DataPointSet
	av = AutomatedViz(table=table, attributes=fields, matcher=dm)
	return await av.retrieve_data_points_2(claim_map, new_attributes, verbose=verbose, info_table=info_table)

@app.post("/get_reason")
async def get_reason(claim_map: ClaimMap, datasets: list[Dataset], verbose:bool=True, fast_mode:bool=True):
	dm = DataMatcher(datasrc="../Datasets")
	claim = claim_map.rephrase
	table, _, _, _, _, _ = dm.merge_datasets(datasets, change_dataset=True)
	tb = TableReasoner(datamatcher=dm)
	if fast_mode:
		return await tb.reason_2(claim_map, table, verbose=verbose, fuzzy_match=True)
	else:
		return await tb.reason(claim, table, verbose=verbose, fuzzy_match=True)

@app.post("/get_datasets")
async def get_relevant_datasets(claim_map: ClaimMap, verbose:bool=True):
	"""
		1. Infer the most related attributes
		2. Infer the @() countries
		3. Infer @() years????
	"""
	value_keywords = [keyword for sublist in claim_map.suggestion for keyword in sublist.values if sublist.field == "value"]
	country_keywords = [keyword[2:-2].replace("Country", "").replace("Countries", "").strip() for keyword in claim_map.country if keyword.startswith("@")]
	keywords = country_keywords + claim_map.value + value_keywords
	print("keywords:", keywords)
	dm = DataMatcher(datasrc="../Datasets")
	tb = TableReasoner(datamatcher=dm)
	top_k_datasets = await dm.find_top_k_datasets("", k=5, method="gpt", verbose=verbose, keywords=keywords)
	datasets = [Dataset(name=name, description=description, score=score, fields=fields) 
        for name, description, score, fields in top_k_datasets]

	# 1. Infer the most related attributes
	table, country_attr, date_attr, fields, embeddings, _ = dm.merge_datasets(datasets)
	attributes = claim_map.value
	scores = cosine_similarity(dm.encode(attributes), embeddings)
	argmax_indices = scores.argmax(axis=1)
	
	warn_flag, warning = False, ""
	for i, score in enumerate(scores):
		if score[argmax_indices[i]] < 0.5:
			warning = f"The pipeline is not confident."
			print(f"{'@'*100}\n{warning}. Score: {score[argmax_indices[i]]}\n{'@'*100}")
			warn_flag = True
			break
	if not warn_flag:
		print(f"{'@'*100}\nThe pipeline is confident. Score: {min(score[argmax_indices[i]] for i, score in enumerate(scores))}\n{'@'*100}")

	new_attributes = [fields[i] for i in argmax_indices] 
	print("new_attributes:", new_attributes)
	claim_map.mapping.update({attr: new_attributes[i] for i, attr in enumerate(attributes)})

	# update date and country real attribute name
	print("Country:", country_attr, "Date:", date_attr)
	claim_map.mapping.update({"date": date_attr, "country": country_attr})
	claim_map.cloze_vis = claim_map.cloze_vis.replace("{date}", f'{{{date_attr}}}').replace("{country}", f'{{{country_attr}}}')

	# 2. Infer the @() countries
	infer_country_tasks, country_to_infer = [], []
	for idx, country in enumerate(claim_map.country):
		if country.startswith('@('):
			if any(p in country for p in ["Bottom", "Top", "with", "Countries of"]):
				infer_country_tasks.append(
					tb._infer_country(
						country[2:-2], claim_map.date, 
						new_attributes, table, datasets
					)
				)	
				country_to_infer.append(country)
			else: # query like @(Asian countries?) have been handled by the _suggest_variable module
				cntry_sets = [cntry_set for cntry_set in claim_map.suggestion if cntry_set.field == "countries"]
				suggest_countries = set(cntry for sublist in cntry_sets for cntry in sublist.values)
				actual_suggest_countries = []
				for cntry in suggest_countries:
					matched_cells = _get_matched_cells(cntry, dm, table, attr=country_attr)
					if matched_cells:
						actual_suggest_countries.append(matched_cells[0][0])
				# suggest_countries = random.sample(suggest_countries, 5)
				claim_map.mapping[country] = actual_suggest_countries[:5] # take the top 5 suggested
		else:
			claim_map.country[idx] = _get_matched_cells(country, dm, table, attr=country_attr)[0][0]
	
	inferred_countries = await asyncio.gather(*infer_country_tasks)
	claim_map.mapping.update({country_to_infer[idx]: country_list for idx, country_list in enumerate(inferred_countries)})

	return {
		"datasets": datasets,
		"claim_map": claim_map,
		"warning": warning
	}

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

@app.post('/dataset_explanation_2') 
def get_dataset_explanation_2(datasets: list[Dataset], column_name: str):
	for dataset in datasets:
		if column_name in dataset.fields:
			table_name = dataset.name
			break
	df = pd.read_csv(f"../Datasets/info/{table_name}")
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
async def get_suggested_queries(claim: UserClaimBody, model: Model = Model.GPT_TAG_4):
	tagged_claim = await TableReasoner()._suggest_queries_2(claim, model=model)
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
	# p = Profiler()
	# p.start()
	paragraph = ""
	userClaim = "the total fertility rate began to sink more quickly in the 2000s during the financial crises. "
	# userClaim = "New Zealand's GDP is 10% from tourism."
	# A significant amount of New Zealand's GDP comes from tourism
	claim = UserClaimBody(userClaim=userClaim, paragraph=paragraph)
	claim_map = await get_suggested_queries(claim, model=Model.GPT_TAG_4)
	print(f"{claim_map}\n{'@'*100}")

# 	claim_map = {
#   "country": [
# 	"World"
#   ],
#   "value": [
# 	"economic indicators",
# 	"financial crises"
#   ],
#   "date": [
# 	"1990 - 1999",
# 	"2000 - 2009"
#   ],
#   "vis": "Show the {total fertility rate} of the {World} from {1990} to {1999} and from {2000} to {2009}.",
#   "cloze_vis": "Show the {value} of the {country} from {date} to {date} and from {date} to {date}.",
#   "rephrase": "",
#   "suggestion": [
# 	{
#   	"field": "value",
#   	"values": [
#     	"economic indicators",
#     	"financial crises"
#   	],
#   	"explain": "How did the financial crises in the 2000s impact other economic indicators in the country?"
# 	},
# 	{
#   	"field": "value",
#   	"values": [
#     	"housing costs",
#     	"child care costs",
#     	"education costs",
#     	"unemployment rate"
#   	],
#   	"explain": "What specific factors contributed to the decline in the total fertility rate during the financial crises?"
# 	},
# 	{
#   	"field": "value",
#   	"values": [
#     	"youth unemployment rate",
#     	"youth anxiety levels"
#   	],
#   	"explain": "How did the financial crises affect the employment prospects and mental well-being of young people in the country?"
# 	},
# 	{
#   	"field": "datetime",
#   	"values": [
#     	"1997",
#     	"2008"
#   	],
#   	"explain": "How did the financial crises in South Korea impact the total fertility rate in the 2000s?"
# 	},
# 	{
#   	"field": "datetime",
#   	"values": [
#     	"@(Year with the lowest total fertility rate)"
#   	],
#   	"explain": "When did South Korea experience the lowest total fertility rate?"
# 	},
# 	{
#   	"field": "country",
#   	"values": [
#     	"South Korea"
#   	],
#   	"explain": "How does the decline in total fertility rate during the financial crises in South Korea compare to other countries?"
# 	},
# 	{
#   	"field": "country",
#   	"values": [
#     	"Japan",
#     	"Germany",
#     	"Italy"
#   	],
#   	"explain": "How did the financial crises impact the total fertility rate in other developed countries?"
# 	},
# 	{
#   	"field": "country",
#   	"values": [
#     	"United States",
#     	"United Kingdom",
#     	"Canada"
#   	],
#   	"explain": "How does the decline in total fertility rate during the financial crises in South Korea compare to other English-speaking countries?"
# 	},
# 	{
#   	"field": "country",
#   	"values": [
#     	"China",
#     	"Taiwan",
#     	"Hong Kong"
#   	],
#   	"explain": "How did the financial crises impact the total fertility rate in other East Asian countries?"
# 	},
# 	{
#   	"field": "country",
#   	"values": [
#     	"@(Countries with the highest fertility rates)"
#   	],
#   	"explain": "Which countries have the highest fertility rates?"
# 	}
#   ],
#   "mapping": {}
# }

# 	claim_map = ClaimMap(**claim_map)

	dic = await get_relevant_datasets(claim_map)
	top_k_datasets, claim_map = dic["datasets"], dic["claim_map"]

	dtps = await potential_data_point_sets_2(claim_map, top_k_datasets)
	print(dtps)
	# p = Profiler()
	# with p:
	reason = await get_reason(claim_map, top_k_datasets, verbose=True)

if __name__ == "__main__":
	asyncio.run(main())