import random
from typing import Annotated
import numpy as np
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Body, Depends, FastAPI, HTTPException
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
from ClaimDetection import ClaimDetector

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
	dm = DataMatcher(datasrc="../Datasets", load_desc=False)
	table, _, _, fields, _, info_table = dm.merge_datasets(datasets, change_dataset=True)
	new_attributes = [claim_map.mapping[attr] for attr in claim_map.value]	

	# 3. return DataPointSet
	av = AutomatedViz(table=table, attributes=fields, matcher=dm)
	return await av.retrieve_data_points_2(claim_map, new_attributes, verbose=verbose, info_table=info_table)

@app.post("/get_reason")
async def get_reason(claim_map: ClaimMap, datasets: list[Dataset], verbose:bool=True, fast_mode:bool=True):
	dm = DataMatcher(datasrc="../Datasets", load_desc=False)
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

	value_keywords = [keyword for sublist in claim_map.suggestion for keyword in sublist.values if sublist.field == "value" or keyword.startswith("@(")]
	country_keywords = [keyword[2:-2].replace("Country", "").replace("Countries", "").strip() for keyword in claim_map.country if keyword.startswith("@(")]
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

	# 2. Infer the @() countries/ @() years from both the claim and the suggested values
	infer_country_tasks, country_to_infer = [], []
	for idx, country in enumerate(claim_map.country):
		if country.startswith('@('):
			if any(p in country for p in ["Bottom", "Top", "with", "Countries of"]):
				infer_country_tasks.append(
					tb._infer_country(
						country[2:-2], claim_map.date, 
						new_attributes, table
					)
				)	
				country_to_infer.append(country)
			else: # query like @(Asian countries?) have been handled by the _suggest_variable module
				cntry_sets = [cntry_set for cntry_set in claim_map.suggestion if cntry_set.field == tb.INDICATOR["countries"]]
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

	for suggest in claim_map.suggestion: 
		for val in suggest.values:
			if val.startswith('@('):
				infer_country_tasks.append(
					tb._infer_country(
						val[2:-2], claim_map.date, 
						new_attributes, table
					)
				)
				country_to_infer.append(val)

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

@app.post("/get_data_2")
def get_data_new_2(body: GetVizDataBodyMulti) -> list[dict]:
	datasets = body.datasets
	df, _, _, _, _, _ = DataMatcher(datasrc="../Datasets", load_desc=False).merge_datasets(datasets)

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
def get_dataset_explanation_2(datasets: list[Dataset], column_name: Annotated[str, Body()]):
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
async def get_suggested_queries(claim: UserClaimBody, model: Model = Model.GPT_TAG_4, verbose:bool=True):
	tb = TableReasoner(datamatcher=DataMatcher(datasrc="../Datasets"))
	tagged_claim = await tb._suggest_queries_2(claim, model=model, verbose=verbose)
	return await tb._get_relevant_datasets(tagged_claim, verbose=verbose)

@app.post('/detect_claim')
async def detect_claim(claim: UserClaimBody):
	detector = ClaimDetector()
	return await detector.detect_2(claim.userClaim)

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
	userClaim = "Vietnam has become the largest coal manufacturer since 2017."
	# userClaim = "New Zealand's GDP is 10% from tourism."
	# A significant amount of New Zealand's GDP comes from tourism
	claim = UserClaimBody(userClaim=userClaim, paragraph=paragraph)
	dic = await get_suggested_queries(claim, model=Model.GPT_TAG_4)
	top_k_datasets, claim_map = dic["datasets"], dic["claim_map"]
	print("claim_map:", claim_map)
	# BodyViz = {
	# 		"datasets":[
	# 			{
	# 				"name":"Health.csv",
	# 				"description":"",
	# 				"score":0.4800550764391787,
	# 				"fields":[
	# 						"Wanted fertility rate (births per woman)","Adolescent fertility rate (births per 1,000 women ages 15-19)","Fertility rate, total (births per woman)","country_name","date"
	# 				]
	# 			},
	# 			{
	# 				"name":"Economy & Growth.csv",
	# 				"description":"",
	# 				"score":0.44914940256599467,
	# 				"fields":["GDP per capita growth (annual %)","GDP growth (annual %)","country_name","date"
	# 				]
	# 			},
	# 			{
	# 				"name":"Gender.csv",
	# 				"description":"",
	# 				"score":0.43736179043032325,
	# 				"fields":[
	# 					"Wanted fertility rate (births per woman)","Adolescent fertility rate (births per 1,000 women ages 15-19)","Fertility rate, total (births per woman)","country_name","date"]
	# 			},
	# 			{
	# 				"name":"Environment.csv",
	# 				"description":"",
	# 				"score":0.4293302021300258,
	# 				"fields":["Adjusted savings: education expenditure (current US$)","country_name","date"]
	# 			},
	# 			{
	# 				"name":"Climate Change.csv",
	# 				"description":"",
	# 				"score":0.4060632659846748,
	# 				"fields":["Population growth (annual %)","country_name","date"]
	# 			}],
	# 		"values":[
	# 			{
	# 				"label":"Droughts, floods, extreme temperatures (% of population, average 1990-2009)",
    #  				"value":"Droughts, floods, extreme temperatures (% of population, average 1990-2009)",
	# 				"unit":"% of population, average 1990-2009",
	# 				"provenance":"EM-DAT: The OFDA/CRED International Disaster Database: www.emdat.be, Université Catholique de Louvain, Brussels (Belgium), World Bank."
	# 			},
	# 			{
	# 				"label":"Droughts, floods, extreme temperatures (% of population, average 1990-2009)",
	# 				"value":"Droughts, floods, extreme temperatures (% of population, average 1990-2009)",
	# 				"unit":"% of population, average 1990-2009","provenance":"EM-DAT: The OFDA/CRED International Disaster Database: www.emdat.be, Université Catholique de Louvain, Brussels (Belgium), World Bank."
	# 			},
	# 			{
	# 				"label":"Adolescent fertility rate (births per 1,000 women ages 15-19)",
	# 				"value":"Adolescent fertility rate (births per 1,000 women ages 15-19)",
	# 				"unit":"births per 1,000 women ages 15-19",
	# 				"provenance":"United Nations Population Division, World Population Prospects."
	# 			},
	# 			{
	# 				"label":"Wanted fertility rate (births per woman)",
	# 				"value":"Wanted fertility rate (births per woman)",
	# 				"unit":"births per woman",
	# 				"provenance":"Demographic and Health Surveys."
	# 			},
	# 			{
	# 				"label":"Mortality rate, under-5 (per 1,000 live births)",
	# 				"value":"Mortality rate, under-5 (per 1,000 live births)",
	# 				"unit":"per 1,000 live births",
	# 				"provenance":"Estimates developed by the UN Inter-agency Group for Child Mortality Estimation (UNICEF, WHO, World Bank, UN DESA Population Division) at www.childmortality.org."
	# 			},
	# 			{
	# 				"label":"Population growth (annual %)",
	# 				"value":"Population growth (annual %)",
	# 				"unit":"annual %",
	# 				"provenance":"Derived from total population. Population source: (1) United Nations Population Division. World Population Prospects: 2022 Revision, (2) Census reports and other statistical publications from national statistical offices, (3) Eurostat: Demographic Statistics, (4) United Nations Statistical Division. Population and Vital Statistics Reprot (various years), (5) U.S. Census Bureau: International Database, and (6) Secretariat of the Pacific Community: Statistics and Demography Programme."
	# 			},
	# 			{
	# 				"label":"Fertility rate, total (births per woman)",
	# 				"value":"Fertility rate, total (births per woman)",
	# 				"unit":"births per woman",
	# 				"provenance":"(1) United Nations Population Division. World Population Prospects: 2022 Revision. (2) Census reports and other statistical publications from national statistical offices, (3) Eurostat: Demographic Statistics, (4) United Nations Statistical Division. Population and Vital Statistics Reprot (various years), (5) U.S. Census Bureau: International Database, and (6) Secretariat of the Pacific Community: Statistics and Demography Programme."
	# 			},
	# 			{
	# 				"label":"Primary completion rate, total (% of relevant age group)",
	# 				"value":"Primary completion rate, total (% of relevant age group)",
	# 				"unit":"% of relevant age group",
	# 				"provenance":"UNESCO Institute for Statistics (UIS). UIS.Stat Bulk Data Download Service. Accessed October 24, 2022. https://apiportal.uis.unesco.org/bdds."
	# 			},
	# 			{
	# 				"label":"Adjusted savings: education expenditure (current US$)",
	# 				"value":"Adjusted savings: education expenditure (current US$)",
	# 				"unit":"current US$",
	# 				"provenance":"World Bank staff estimates using data from the United Nations Statistics Division's Statistical Yearbook, and the UNESCO Institute for Statistics online database."
	# 			}],
	# 		"fields":{
	# 			"country_name":[
	# 				{"label":"Afghanistan","value":"Afghanistan"},
	# 				{"label":"Africa Eastern and Southern","value":"Africa Eastern and Southern"},
	# 				{"label":"Africa Western and Central","value":"Africa Western and Central"},{"label":"Albania","value":"Albania"},{"label":"Algeria","value":"Algeria"},{"label":"American Samoa","value":"American Samoa"},{"label":"Andorra","value":"Andorra"},{"label":"Angola","value":"Angola"},{"label":"Antigua and Barbuda","value":"Antigua and Barbuda"},{"label":"Arab World","value":"Arab World"},{"label":"Argentina","value":"Argentina"},{"label":"Armenia","value":"Armenia"},{"label":"Aruba","value":"Aruba"},{"label":"Australia","value":"Australia"},{"label":"Austria","value":"Austria"},{"label":"Azerbaijan","value":"Azerbaijan"},{"label":"Bahamas, The","value":"Bahamas, The"},{"label":"Bahrain","value":"Bahrain"},{"label":"Bangladesh","value":"Bangladesh"},{"label":"Barbados","value":"Barbados"},{"label":"Belarus","value":"Belarus"},{"label":"Belgium","value":"Belgium"},{"label":"Belize","value":"Belize"},{"label":"Benin","value":"Benin"},{"label":"Bermuda","value":"Bermuda"},{"label":"Bhutan","value":"Bhutan"},{"label":"Bolivia","value":"Bolivia"},{"label":"Bosnia and Herzegovina","value":"Bosnia and Herzegovina"},{"label":"Botswana","value":"Botswana"},{"label":"Brazil","value":"Brazil"},{"label":"British Virgin Islands","value":"British Virgin Islands"},{"label":"Brunei Darussalam","value":"Brunei Darussalam"},{"label":"Bulgaria","value":"Bulgaria"},{"label":"Burkina Faso","value":"Burkina Faso"},{"label":"Burundi","value":"Burundi"},{"label":"Cabo Verde","value":"Cabo Verde"},{"label":"Cambodia","value":"Cambodia"},{"label":"Cameroon","value":"Cameroon"},{"label":"Canada","value":"Canada"},{"label":"Caribbean small states","value":"Caribbean small states"},{"label":"Cayman Islands","value":"Cayman Islands"},{"label":"Central African Republic","value":"Central African Republic"},{"label":"Central Europe and the Baltics","value":"Central Europe and the Baltics"},{"label":"Chad","value":"Chad"},{"label":"Channel Islands","value":"Channel Islands"},{"label":"Chile","value":"Chile"},{"label":"China","value":"China"},{"label":"Colombia","value":"Colombia"},{"label":"Comoros","value":"Comoros"},{"label":"Costa Rica","value":"Costa Rica"},{"label":"Cote d'Ivoire","value":"Cote d'Ivoire"},{"label":"Croatia","value":"Croatia"},{"label":"Cuba","value":"Cuba"},{"label":"Curacao","value":"Curacao"},{"label":"Cyprus","value":"Cyprus"},{"label":"Czechia","value":"Czechia"},{"label":"Democratic People's Republic of Korea","value":"Democratic People's Republic of Korea"},{"label":"Democratic Republic of Congo","value":"Democratic Republic of Congo"},{"label":"Denmark","value":"Denmark"},{"label":"Djibouti","value":"Djibouti"},{"label":"Dominica","value":"Dominica"},{"label":"Dominican Republic","value":"Dominican Republic"},{"label":"Early-demographic dividend","value":"Early-demographic dividend"},{"label":"East Asia & Pacific","value":"East Asia & Pacific"},{"label":"East Asia & Pacific (excluding high income)","value":"East Asia & Pacific (excluding high income)"},{"label":"East Asia & Pacific (IDA & IBRD countries)","value":"East Asia & Pacific (IDA & IBRD countries)"},{"label":"Ecuador","value":"Ecuador"},{"label":"Egypt, Arab Rep.","value":"Egypt, Arab Rep."},{"label":"El Salvador","value":"El Salvador"},{"label":"Equatorial Guinea","value":"Equatorial Guinea"},{"label":"Eritrea","value":"Eritrea"},{"label":"Estonia","value":"Estonia"},{"label":"Eswatini","value":"Eswatini"},{"label":"Ethiopia","value":"Ethiopia"},{"label":"Euro area","value":"Euro area"},{"label":"Europe & Central Asia","value":"Europe & Central Asia"},{"label":"Europe & Central Asia (excluding high income)","value":"Europe & Central Asia (excluding high income)"},{"label":"Europe & Central Asia (IDA & IBRD countries)","value":"Europe & Central Asia (IDA & IBRD countries)"},{"label":"European Union","value":"European Union"},{"label":"Faroe Islands","value":"Faroe Islands"},{"label":"Fiji","value":"Fiji"},{"label":"Finland","value":"Finland"},{"label":"Fragile and conflict affected situations","value":"Fragile and conflict affected situations"},{"label":"France","value":"France"},{"label":"French Polynesia","value":"French Polynesia"},{"label":"Gabon","value":"Gabon"},{"label":"Gambia, The","value":"Gambia, The"},{"label":"Georgia","value":"Georgia"},{"label":"Germany","value":"Germany"},{"label":"Ghana","value":"Ghana"},{"label":"Gibraltar","value":"Gibraltar"},{"label":"Greece","value":"Greece"},{"label":"Greenland","value":"Greenland"},{"label":"Grenada","value":"Grenada"},{"label":"Guam","value":"Guam"},{"label":"Guatemala","value":"Guatemala"},{"label":"Guinea","value":"Guinea"},{"label":"Guinea-Bissau","value":"Guinea-Bissau"},{"label":"Guyana","value":"Guyana"},{"label":"Haiti","value":"Haiti"},{"label":"Heavily indebted poor countries (HIPC)","value":"Heavily indebted poor countries (HIPC)"},{"label":"High income","value":"High income"},{"label":"Honduras","value":"Honduras"},{"label":"Hong Kong SAR, China","value":"Hong Kong SAR, China"},{"label":"Hungary","value":"Hungary"},{"label":"IBRD only","value":"IBRD only"},{"label":"Iceland","value":"Iceland"},{"label":"IDA & IBRD total","value":"IDA & IBRD total"},{"label":"IDA blend","value":"IDA blend"},{"label":"IDA only","value":"IDA only"},{"label":"IDA total","value":"IDA total"},{"label":"India","value":"India"},{"label":"Indonesia","value":"Indonesia"},{"label":"Iran, Islamic Rep.","value":"Iran, Islamic Rep."},{"label":"Iraq","value":"Iraq"},{"label":"Ireland","value":"Ireland"},{"label":"Isle of Man","value":"Isle of Man"},{"label":"Israel","value":"Israel"},{"label":"Italy","value":"Italy"},{"label":"Jamaica","value":"Jamaica"},{"label":"Japan","value":"Japan"},{"label":"Jordan","value":"Jordan"},{"label":"Kazakhstan","value":"Kazakhstan"},{"label":"Kenya","value":"Kenya"},{"label":"Kiribati","value":"Kiribati"},{"label":"Kosovo","value":"Kosovo"},{"label":"Kuwait","value":"Kuwait"},{"label":"Kyrgyz Republic","value":"Kyrgyz Republic"},{"label":"Lao PDR","value":"Lao PDR"},{"label":"Late-demographic dividend","value":"Late-demographic dividend"},{"label":"Latin America & Caribbean","value":"Latin America & Caribbean"},{"label":"Latin America & Caribbean (excluding high income)","value":"Latin America & Caribbean (excluding high income)"},{"label":"Latin America & the Caribbean (IDA & IBRD countries)","value":"Latin America & the Caribbean (IDA & IBRD countries)"},{"label":"Latvia","value":"Latvia"},{"label":"Least developed countries: UN classification","value":"Least developed countries: UN classification"},{"label":"Lebanon","value":"Lebanon"},{"label":"Lesotho","value":"Lesotho"},{"label":"Liberia","value":"Liberia"},{"label":"Libya","value":"Libya"},{"label":"Liechtenstein","value":"Liechtenstein"},{"label":"Lithuania","value":"Lithuania"},{"label":"Low & middle income","value":"Low & middle income"},{"label":"Low income","value":"Low income"},{"label":"Lower middle income","value":"Lower middle income"},{"label":"Luxembourg","value":"Luxembourg"},{"label":"Macao SAR, China","value":"Macao SAR, China"},{"label":"Madagascar","value":"Madagascar"},{"label":"Malawi","value":"Malawi"},{"label":"Malaysia","value":"Malaysia"},{"label":"Maldives","value":"Maldives"},{"label":"Mali","value":"Mali"},{"label":"Malta","value":"Malta"},{"label":"Marshall Islands","value":"Marshall Islands"},{"label":"Mauritania","value":"Mauritania"},{"label":"Mauritius","value":"Mauritius"},{"label":"Mexico","value":"Mexico"},{"label":"Micronesia, Fed. Sts.","value":"Micronesia, Fed. Sts."},{"label":"Middle East & North Africa","value":"Middle East & North Africa"},{"label":"Middle East & North Africa (excluding high income)","value":"Middle East & North Africa (excluding high income)"},{"label":"Middle East & North Africa (IDA & IBRD countries)","value":"Middle East & North Africa (IDA & IBRD countries)"},{"label":"Middle income","value":"Middle income"},{"label":"Moldova","value":"Moldova"},{"label":"Monaco","value":"Monaco"},{"label":"Mongolia","value":"Mongolia"},{"label":"Montenegro","value":"Montenegro"},{"label":"Morocco","value":"Morocco"},{"label":"Mozambique","value":"Mozambique"},{"label":"Myanmar","value":"Myanmar"},{"label":"Namibia","value":"Namibia"},{"label":"Nauru","value":"Nauru"},{"label":"Nepal","value":"Nepal"},{"label":"Netherlands","value":"Netherlands"},{"label":"New Caledonia","value":"New Caledonia"},{"label":"New Zealand","value":"New Zealand"},{"label":"Nicaragua","value":"Nicaragua"},{"label":"Niger","value":"Niger"},{"label":"Nigeria","value":"Nigeria"},{"label":"North America","value":"North America"},{"label":"North Macedonia","value":"North Macedonia"},{"label":"Northern Mariana Islands","value":"Northern Mariana Islands"},{"label":"Norway","value":"Norway"},{"label":"Not classified","value":"Not classified"},{"label":"OECD members","value":"OECD members"},{"label":"Oman","value":"Oman"},{"label":"Other small states","value":"Other small states"},{"label":"Pacific island small states","value":"Pacific island small states"},{"label":"Pakistan","value":"Pakistan"},{"label":"Palau","value":"Palau"},{"label":"Panama","value":"Panama"},{"label":"Papua New Guinea","value":"Papua New Guinea"},{"label":"Paraguay","value":"Paraguay"},{"label":"Peru","value":"Peru"},{"label":"Philippines","value":"Philippines"},{"label":"Poland","value":"Poland"},{"label":"Portugal","value":"Portugal"},{"label":"Post-demographic dividend","value":"Post-demographic dividend"},{"label":"Pre-demographic dividend","value":"Pre-demographic dividend"},{"label":"Puerto Rico","value":"Puerto Rico"},{"label":"Qatar","value":"Qatar"},{"label":"Republic of Congo","value":"Republic of Congo"},{"label":"Republic of Korea","value":"Republic of Korea"},{"label":"Romania","value":"Romania"},{"label":"Russian Federation","value":"Russian Federation"},{"label":"Rwanda","value":"Rwanda"},{"label":"Samoa","value":"Samoa"},{"label":"San Marino","value":"San Marino"},{"label":"Sao Tome and Principe","value":"Sao Tome and Principe"},{"label":"Saudi Arabia","value":"Saudi Arabia"},{"label":"Senegal","value":"Senegal"},{"label":"Serbia","value":"Serbia"},{"label":"Seychelles","value":"Seychelles"},{"label":"Sierra Leone","value":"Sierra Leone"},{"label":"Singapore","value":"Singapore"},{"label":"Sint Maarten (Dutch part)","value":"Sint Maarten (Dutch part)"},{"label":"Slovak Republic","value":"Slovak Republic"},{"label":"Slovenia","value":"Slovenia"},{"label":"Small states","value":"Small states"},{"label":"Solomon Islands","value":"Solomon Islands"},{"label":"Somalia","value":"Somalia"},{"label":"South Africa","value":"South Africa"},{"label":"South Asia","value":"South Asia"},{"label":"South Asia (IDA & IBRD)","value":"South Asia (IDA & IBRD)"},{"label":"South Sudan","value":"South Sudan"},{"label":"Spain","value":"Spain"},{"label":"Sri Lanka","value":"Sri Lanka"},{"label":"St. Kitts and Nevis","value":"St. Kitts and Nevis"},{"label":"St. Lucia","value":"St. Lucia"},{"label":"St. Martin (French part)","value":"St. Martin (French part)"},{"label":"St. Vincent and the Grenadines","value":"St. Vincent and the Grenadines"},{"label":"Sub-Saharan Africa","value":"Sub-Saharan Africa"},{"label":"Sub-Saharan Africa (excluding high income)","value":"Sub-Saharan Africa (excluding high income)"},{"label":"Sub-Saharan Africa (IDA & IBRD countries)","value":"Sub-Saharan Africa (IDA & IBRD countries)"},{"label":"Sudan","value":"Sudan"},{"label":"Suriname","value":"Suriname"},{"label":"Sweden","value":"Sweden"},{"label":"Switzerland","value":"Switzerland"},{"label":"Syrian Arab Republic","value":"Syrian Arab Republic"},{"label":"Tajikistan","value":"Tajikistan"},{"label":"Tanzania","value":"Tanzania"},{"label":"Thailand","value":"Thailand"},{"label":"Timor-Leste","value":"Timor-Leste"},{"label":"Togo","value":"Togo"},{"label":"Tonga","value":"Tonga"},{"label":"Trinidad and Tobago","value":"Trinidad and Tobago"},{"label":"Tunisia","value":"Tunisia"},{"label":"Turkiye","value":"Turkiye"},{"label":"Turkmenistan","value":"Turkmenistan"},{"label":"Turks and Caicos Islands","value":"Turks and Caicos Islands"},{"label":"Tuvalu","value":"Tuvalu"},{"label":"Uganda","value":"Uganda"},{"label":"Ukraine","value":"Ukraine"},{"label":"United Arab Emirates","value":"United Arab Emirates"},{"label":"United Kingdom","value":"United Kingdom"},{"label":"United States","value":"United States"},{"label":"Upper middle income","value":"Upper middle income"},{"label":"Uruguay","value":"Uruguay"},{"label":"Uzbekistan","value":"Uzbekistan"},{"label":"Vanuatu","value":"Vanuatu"},{"label":"Venezuela, RB","value":"Venezuela, RB"},{"label":"Vietnam","value":"Vietnam"},{"label":"Virgin Islands (U.S.)","value":"Virgin Islands (U.S.)"},{"label":"West Bank and Gaza","value":"West Bank and Gaza"},{"label":"World","value":"World"},{"label":"Yemen, Rep.","value":"Yemen, Rep."},{"label":"Zambia","value":"Zambia"},{"label":"Zimbabwe","value":"Zimbabwe"}
	# 			],
	# 			"date":{
	# 				"date_start":{"label":"1960","value":"1960","unit":None,"provenance":None},
	# 				"date_end":{"label":"2020","value":"2020","unit":None,"provenance":None}
	# 			}
	# 		}
	# 	}

	# dts = get_data_new_2(GetVizDataBodyMulti(**BodyViz))
	# print(dts)

	import copy
	dtps = await potential_data_point_sets_2(claim_map, copy.deepcopy(top_k_datasets))
	print(dtps)
	# p = Profiler()
	# with p:
	# reason = await get_reason(claim_map, top_k_datasets, verbose=True)

if __name__ == "__main__":
	asyncio.run(main())