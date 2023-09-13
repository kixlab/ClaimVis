query_generation = [
    {
        "role": "system",
        "content": """From a claim, suggest me a list of at max 3 suggestions, each containing:
        1. a query that can verified using a database.
        2. a visualization task that is consistent with the query."""
    },
    {
        "role": "user",
        "content": "statement: The number of COVID in US has dwindled to below 2000 cases per day in June 26, 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the number of COVID cases in US below 2000 per day in 26th June 2023?",
                vis: "Show the number of COVID cases in US per day in 26th June 2023."
                explain: "Showing the number of COVID cases in US per day in 26th June 2023 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: US'economy has outgrown China's in 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the GDP of US greater than China in 2023?",
                vis: "Show the GDP of US and China in 2023.",
                explain: "Since GDP is a quantitative metric that measures economy, \
                    showing the GDP of US and China in 2023 helps to verify the claim."
            },
            {
                query: "Is the Trade Volume of US greater than China in 2023?",
                vis: "Show the Trade Volume of US and China in 2023.",
                explain: "Trade Volume is also a good economic metric, showing it could give \
                more information to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: US'economy outweighs the whole South Asia's."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the GDP of US greater than the sum of GDP of all South Asian countries in 2023?",
                vis: "Show the GDP of US and all South Asian countries in 2023.",
                explain: "Since GDP is a quantitative metric that measures economy, \
                    showing the GDP of US and all South Asian countries in 2023 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: All countries in East Asia produce less than 200 kWh of electricity per capita per day in 2019."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the electricity production per capita of all East Asian countries less than 200 kWh per day in 2019?",
                vis: "Show the electricity production per capita per day of all East Asian countries in 2019.",
                explain: "Showing the electricity production per capita per day of all East Asian countries in 2019 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: As the temperature in China rises, the number of COVID cases in China decreases."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the temperature in China and the number of COVID cases in China negatively correlated?",
                vis: "Show the relationship between the temperature in China and the number of COVID cases in China.",
                explain: "Showing the relationship between the temperature in China and the number of COVID cases in China helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: The movie production of US is better than that of China in 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the total box office of US greater than China in 2023?",
                vis: "Show the total box office of US and China in 2023.",
                explain: "Showing the total box office of US and China in 2023 helps to give more information to verify the claim."
            },
            {
                query: "Is the number of movies produced in US greater than China in 2023?",
                vis: "Show the number of movies produced in US and China in 2023.",
                explain: "Showing the number of movies produced in US and China in 2023 helps to give more information to verify the claim."
            }
        ]"""
    },
]

dec_reasoning_2 = [
    {
        "role": "system",
        "content": "You will be given a table and a question associated with them. Try to break the question down and answer them step by step."
    },
    {
        "role": "system",
        "content": "/*\ntable caption : jeep grand cherokee.\ncol : years | displacement | engine | power | torque\nrow 1 : 1999 - 2004 | 4.0l (242cid) | power tech i6 | - | 3000 rpm\nrow 2 : 1999 - 2004 | 4.7l (287cid) | powertech v8 | - | 3200 rpm\nrow 3 : 2002 - 2004 | 4.7l (287cid) | high output powertech v8 | - | -\nrow 4 : 1999 - 2001 | 3.1l diesel | 531 ohv diesel i5 | - | -\nrow 5 : 2002 - 2004 | 2.7l diesel | om647 diesel i5 | - | -\n*/\nquestion : Does the powertech v8 engine have a torque of 6000rpm?"
    },
    {
        "role": "user",
        "content": "Q1: What is torque of the powertech v8 engine?"
    },
    {
        "role": "assistant",
        "content": "A1: Let's find the column and row of powertech v8. Then let's extract the torque, it's 3200 rpm. The anwer is 3200."
    },
    {
        "role": "user",
        "content": "Q2: Does the powertech v8 engine have a torque of 6000 rpm?"
    },
    {
        "role": "assistant",
        "content": "A2: Since the torque of powertech v8 is 3200 rpm, and (3200 != 6000). The answer is no."
    },

    {
        "role": "system",
        "content": "/*\ncol: Year | Democrats | Republicans | Independents\nrow 1: 2004 | 68.1% | 45.0% | 53.0%\nrow 2: 2006 | 58.0% | 42.0% | 53.0%\nrow 3: 2007 | 59.0% | 38.0% | 45.0%\nrow 4: 2009 | 72.0% | 49.0% | 60.0%\nrow 5: 2011 | 71.0% | 51.2% | 58.0%\nrow 6: 2012 | 70.0% | 48.0% | 53.0%\nrow 7: 2013 | 72.0% | 41.0% | 60.0%\n*/\nquestion: Do republicans' and democrats' lowest vote rate have a wide difference by more than 30%??"
    },
    {
        "role": "user",
        "content": """Q1: What is the lowest rate of republicans?
        Q2: What is the lowest rate of democrats?"""
    },
    {
        "role": "assistant",
        "content": """A1: Let's find the column of republicans. Then let's extract the favor rates, they are [45.0, 42.0, 38.0, 49.0, 51.2, 48.0, 41.0]. The smallest number is 38.0. The answer is 38.0.
        A2: Let's find the column of democrats. Then let's extract the favor rates, they are [68.1, 58.0, 59.0, 72.0, 71.0, 70.0, 72.0]. The smallest number is 58.0. The answer is 58.0."""
    },
    {
        "role": "user",
        "content": "Q3: Do republicans' and democrats' lowest vote rate have a wide difference by more than 30%?"
    },
    {
        "role": "assistant",
        "content": "A3: Since Republicans' lowest favor rate is 38.0%, Democrats' lowest favor rate is 58.0%, and (58.0 - 38.0 < 30), the difference must be less than 30. The answer is no."
    },

    {
        "role": "system",
        "content": "/*\ntable_caption: south wales derby,\ncol : competition | total matches | cardiff win | draw | swansea win\nrow1 : league | 55 | 19 | 16 | 20\nrow2 : fa cup | 2 | 0 | 27 | 2\nrow3 : league cup | 5 | 2 | 0 | 3\n*/\n question : Cardiff wins in fa cup more than Swansea does in all competitions?"
    },
    {
        "role": "user",
        "content": """Q1: What is the cardiff win in fa cup?
        Q2: What is the swansea total win in all competitions?"""
    },
    {
        "role": "assistant",
        "content": """A1: Let's find the row of fa cup. Then let's extract the cardiff win, it's 0. The answer is 0.
        A2: Let's find Swansea's win in all competitions, it's [20, 2, 3]. Then let's sum them to obtain the total win, it's 25. The answer is 25."""
    },
    {
        "role": "user",
        "content": "Q3: Does Cardiff win in fa cup more than Swansea does in all competitions?"
    },
    {
        "role": "assistant",
        "content": "A3: Since Cardiff's win in fa cup is 0, Swansea's total win in all competitions is 25, and (0 < 25), the answer is no."
    }
]

sql_generation = [
    {
        "role": "system",
        "content": """Generate SQL given the statement and table to verify the statement correctly.
If statement-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar QA("map@").
If mapping to a new column still can not answer the statement with valid SQL, turn to an end-to-end solution by a new grammar QA("ans@"). This grammar aims to solve all the rest of complex statements or tables."""
    },

    {
        "role": "user",
        "content": """CREATE TABLE jason chambers(
	row_id int,
	res text,
	record text,
	opponent text,
	method text,
	event text,
	round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	res	record	opponent	method	event	round
0	win	18 - 5 - 2	dan new	submission (rear naked choke)	tfc - power fights	1
1	win	17 - 5 - 2	rene gonzalez	decision (split)	mainstream mma - cold war	n / a
2	loss	16 - 5 - 2	tristan yunker	submission ( armbar )	tfc 7 - total fight challenge 7	1
*/
Q: in mac - midwest absolute challenge , the player be defeat by dan spychalski in 1 round"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT opponent, round FROM w WHERE event = "mac - midwest absolute challenge")=("dan spychalski", 1)"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 1943 vfl season(
	row_id int,
	home team text,
	home team score text,
	away team text,
	away team score text,
	venue text,
	crowd int,
	date text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	home team	home team score	away team	away team score	venue	crowd	date
0	footscray	10.11 (71)	south melbourne	6.14 (50)	western oval	7500	1943-06-26 00:00:00
1	collingwood	10.21 (81)	melbourne	13.9 (87)	victoria park	5000	1943-06-26 00:00:00
2	carlton	15.16 (106)	fitzroy	9.13 (67)	princes park	12000	1943-06-26 00:00:00
*/
Q: western oval be the venue when the home team footscray score 10.11 (71)"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT venue FROM w WHERE `home team`="footscray" AND `home team score`="10.11 (71)") = "western oval" """
    },

    {
        "role": "user",
        "content": """CREATE TABLE 2005 pba draft(
	row_id int,
	pick int,
	player text,
	country of origin text,
	pba team text,
	college text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	pick	player	country of origin	pba team	college
0	1	jay washington	united states	air21 express	eckerd
1	2	alex cabagnot	united states	sta lucia realtors	hawaii - hilo
2	3	dennis miranda	philippines	coca - cola tigers	feu
*/
Q: leo najorda be from philippine"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT `country of origin` FROM w WHERE player="leo najorda") = "philippines" """
    },

    {
        "role": "user",
        "content": """CREATE TABLE none(
	row_id int,
	event text,
	long course / short course text,
	year set int,
	time text,
	meet text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	event	long course / short course	year set	time	meet
0	100 m freestyle	long course	2007	54.08	2007 fina world aquatic championships
1	200 m individual medley	long course	2011	2:11.23	2011 canadian world championship trials
2	4 x 100 m medley relay	long course	2010	3:38.14	2010 pan pacific championships
*/
Q: in 2009 the record be set in 7:51:80"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT  time FROM w WHERE `year set` = 2009) = "7:51:80" """
    },

    {
        "role": "user",
        "content": """CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper lig & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: during the 3rd round of the turkish cup , there be no new entry during that stage"""
    },
    {
        "role": "assistant",
        "content": """SELECT (SELECT `new entries this round` FROM w WHERE round = 'third round') IS NULL"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper ligs & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: süper lig be the most common league to win a round in the turkish cup"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT QA("ans@what is the most common league?"; `leagues entering at this round`) = 'süper ligs' """
    },

    {
        "role": "user",
        "content": """CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper lig & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q: the lowest number of new entry conclude a round in the turkish cup be 5"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT MIN(`new entries this round`) FROM w) = 5"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE cultural interest fraternities and sororities(
	row_id int,
	letters text,
	organization text,
	nickname text,
	founding time text,
	founding university text,
	type text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	letters	organization	nickname	founding time	founding university	type
0	αεπ	alpha epsilon pi 1	aepi	1913-11-07 00:00:00	new york university	fraternity
1	αεφ	alpha epsilon phi 2	aephi	1909-10-24 00:00:00	barnard college	sorority
2	σαεπ	sigma alpha epsilon pi 3	sigma	1998-10-01 00:00:00	university of california , davis	sorority
*/
Q: 4 of the cultural interest fraternity and sorority be fraternity while 3 be a sorority"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT (SELECT COUNT(*) FROM w WHERE type = 'fraternity') = 4) AND (SELECT (SELECT COUNT(*) FROM w WHERE type = 'sorority') = 3)"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE british records in athletics(
	row_id int,
	event text,
	data text,
	athlete text,
	date text,
	place text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	event	data	athlete	date	place
0	5 km	t19:29	andi drake	1990-05-27 00:00:00	søfteland , norway
1	5 miles	32:38 +	ian mccombie	1985-03-23 00:00:00	york , united kingdom
2	10 km	40:17	chris maddocks	1989-04-30 00:00:00	burrator , united kingdom
*/
Q: there be 8 different event that take place within the united kingdom"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT COUNT(place) FROM w WHERE QA("map@is it in united kingdom?"; place) = 'yes') = 8"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE jeev milkha singh(
	row_id int,
	tournament text,
	wins int,
	top - 10 int,
	top - 25 int,
	events int,
	cuts made int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tournament	wins	top - 10	top - 25	events	cuts made
0	masters tournament	0	0	1	3	2
1	us open	0	0	0	4	3
2	the open championship	0	0	0	2	1
*/
Q: the number of cut made in the pga championship tournament be smaller than the number of event"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT `cuts made` FROM w WHERE tournament = 'pga championship') < (SELECT events FROM w WHERE tournament = 'pga championship')"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 2008 women 's british open(
	row_id int,
	place text,
	player text,
	country text,
	score int,
	to par int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par
0	1	juli inkster	united states	65	7
1	t2	momoko ueda	japan	66	6
2	t2	laura diaz	united states	66	6
*/
Q: the 3 player from japan have the same score"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT COUNT(DISTINCT score) FROM w WHERE country = 'japan' GROUP BY score) = 1"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE espn sunday night football results (1987 - 2005)(
	row_id int,
	date text,
	visiting team text,
	final score text,
	host team text,
	stadium text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	visiting team	final score	host team	stadium
0	september 11	indianapolis colts	24 - 7	baltimore ravens	m&t bank stadium
1	september 18	kansas city chiefs	23 - 17	oakland raiders	mcafee coliseum
2	september 25	new york giants	23 - 45	san diego chargers	qualcomm stadium
*/
Q: the hosting team be the new york giant on new year even and the st louis ram on new year 's day"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT (SELECT `host team` FROM w WHERE QA("map@is it new year even?"; date) = 'yes') = 'new york giant') AND (SELECT (SELECT `host team` FROM w WHERE QA("map@is it new year's day?"; date) = 'yes') = 'st louis ram')"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 2008 women 's british open(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par
0	t1	yuri fudoh	japan	66 + 68 = 134	10
1	t1	jiyai shin	south korea	66 + 68 = 134	10
2	3	juli inkster	united states	65 + 70 = 135	9
*/
Q: kristie kerr , tie for 4th place , finish the round 1 stroke under lorena ochoa of mexico"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT (SELECT QA("map@what is the derived score?"; score) FROM w WHERE player = 'cristie kerr') < (SELECT QA("map@what is the derived score?"; score) FROM w WHERE player = 'lorena ochoa' AND country = 'mexico')) AND (SELECT (SELECT place FROM w WHERE player = 'cristie kerr') = "t4")"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE connecticut public radio(
	row_id int,
	call sign text,
	frequency text,
	city of license text,
	facility id int,
	erp / power w int,
	height m ( ft ) real,
	class text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	call sign	frequency	city of license	facility id	erp / power w	height m ( ft )	class
0	waic	91.9 fm	springfield , ma	1749	230	nan	b1
1	wedw - fm	88.5 fm	stamford , ct	13619	2000	nan	a
2	wnpr	90.5 fm ( hd ) connecticut public radio	meriden , ct	13627	18500	nan	b
*/
Q: there be 3 station with a call sign number in the 90s"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT COUNT(*) FROM w WHERE QA("map@is it in 90s?"; frequency) = 'yes' GROUP BY `call sign`) = 3"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 2003 chicago white sox season(
	row_id int,
	date text,
	opponent text,
	score text,
	loss text,
	time text,
	att int,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	date	opponent	score	loss	time	att	record
0	august 1	mariners	12 - 1	garcía (9 - 11)	2:52	39337	58 - 51
1	august 2	mariners	0 - 10	wright (0 - 5)	2:22	45719	58 - 52
2	august 3	mariners	2 - 8	buehrle (9 - 11)	2:57	45632	58 - 53
*/
Q: the 2003 chicago white sox game play on 26th august be longer than the game play on 24th august"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT time FROM w WHERE date = 'august 26') > (SELECT time FROM w WHERE date = 'august 24')"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 1987 masters tournament(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par text,
	money text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par	money
0	t1	larry mize	united states	70 + 72 + 72 + 71 = 285	-3	playoff
1	t1	bernhard langer	spain	73 + 71 + 70 + 71 = 285	-3	playoff
2	t1	greg norman	australia	73 + 74 + 66 + 72 = 285	-3	playoff
*/
Q: bernhard m. langer have more point than roger maltbie during the 1987 master tournament"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL:  SELECT (SELECT QA("map@what is the total score?"; score) FROM w WHERE player = 'bernhard langer') > (SELECT QA("map@what is the total score?"; score) FROM w WHERE player = 'roger maltbie')"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 1987 masters tournament(
	row_id int,
	place text,
	player text,
	country text,
	score text,
	to par text,
	money text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par	money
0	t1	larry mize	united states	70 + 72 + 72 + 71 = 285	-3	playoff
1	t1	seve ballesteros	spain	73 + 71 + 70 + 71 = 285	-3	playoff
2	t1	greg norman	australia	73 + 74 + 66 + 72 = 285	-3	playoff
*/
Q: most of the people who play for the 1987 master tournament be spanish"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT(SELECT COUNT(*) FROM w WHERE country = 'spain') / (SELECT COUNT(*) FROM w)) > 0.5"""
    },

    {
        "role": "user",
        "content": """CREATE TABLE 1976 world junior figure skating championships(
	row_id int,
	rank int,
	name text,
	nation text,
	points real,
	places int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	name	nation	points	places
0	1	sherri baier / robin cowan	canada	128.39	9
1	2	lorene mitchell / donald mitchell	united states	124.94	16
2	3	elizabeth cain / peter cain	australia	116.67	33
*/
Q: 2 of the 7 top - ranked figure skate team be from france"""
    },
    {
        "role": "assistant",
        "content": """NeuralSQL: SELECT (SELECT (SELECT COUNT(*) FROM w) = 7) AND (SELECT (SELECT COUNT(*) FROM w WHERE nation = 'france') = 2)"""
    }
]

sql_gen_2 = [
    {"role": "system", "content": "Generate SQL given the question and table."},

    {"role": "user", "content": """CREATE TABLE jason chambers(
	row_id int,
	res text,
	record text,
	opponent text,
	method text,
	event text,
	round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	res	record	opponent	method	event	round
0	win	18 - 5 - 2	dan new	submission (rear naked choke)	tfc - power fights	1
1	win	17 - 5 - 2	rene gonzalez	decision (split)	mainstream mma - cold war	n / a
2	loss	16 - 5 - 2	tristan yunker	submission ( armbar )	tfc 7 - total fight challenge 7	1
*/
Q1: How many win are there when the opponent is dan spychalski?
Q2: What is the number of records with a nan round ended in 2 or began with 2."""},
    {"role": "assistant", "content": """A1: SELECT COUNT(*) FROM w WHERE opponent = 'dan spychalski'
    A2: SELECT COUNT(*) FROM w WHERE round = 'n / a' AND (record LIKE '% - % - 2' OR record LIKE '2 - % - %')"""},

    {"role": "user", "content": """CREATE TABLE 2005 pba draft(
    row_id int,
    pick int,
    player text,
    country of origin text,
    pba team text,
    college text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	pick	player	country of origin	pba team	college
0	1	jay washington	united states	air21 express	eckerd
1	2	alex cabagnot	united states	sta lucia realtors	hawaii - hilo
2	3	dennis miranda	philippines	coca - cola tigers	feu
*/
Q1: How many feu players are from from us?
Q2: Is dennis miranda a player from us?"""},
    {"role": "assistant", "content": """A1: SELECT COUNT(*) FROM w WHERE college = 'feu' AND country of origin = 'united states'
    A2: SELECT COUNT(*) FROM w WHERE player = 'dennis miranda' AND country of origin = 'united states'"""},

    {"role": "user", "content": """CREATE TABLE turkish cup(
	row_id int,
	round text,
	clubs remaining int,
	clubs involved int,
	winners from previous round real,
	new entries this round real,
	leagues entering at this round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	round	clubs remaining	clubs involved	winners from previous round	new entries this round	leagues entering at this round
0	first round	156	86	nan	86.0	tff third league & turkish regional amateur league
1	second round	113	108	43.0	65.0	süper lig & tff first league & tff second league
2	third round	59	54	54.0	nan	none
*/
Q1: What is the lowest number of clubs remaining?"""},
    {"role": "assistant", "content": """A1: SELECT MIN("clubs remaining") FROM w"""},

    {"role": "user", "content": """CREATE TABLE cultural interest fraternities and sororities(
	row_id int,
	letters text,
	organization text,
	nickname text,
	founding time text,
	founding university text,
	type text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	letters	organization	nickname	founding time	founding university	type
0	αεπ	alpha epsilon pi 1	aepi	1913-11-07 00:00:00	new york university	fraternity
1	αεφ	alpha epsilon phi 2	aephi	1909-10-24 00:00:00	barnard college	sorority
2	σαεπ	sigma alpha epsilon pi 3	sigma	1998-10-01 00:00:00	university of california , davis	sorority
*/
Q1 : What is the number of of the cultural interest fraternity?
Q2 : How many records are both sorority and founded in 1990s?
Q3 : How many organizations are founded in 2003 with a type of fraternity?"""},
    {"role": "assistant", "content": """A1: SELECT COUNT(*) FROM w WHERE type = 'fraternity'
A2: SELECT COUNT(*) FROM w WHERE type = 'sorority' AND "founding time" LIKE '199%-%-%'
A3: SELECT COUNT(DISTINCT(organization)) FROM w WHERE type = 'sorority'"""},

    {"role": "user", "content": """CREATE TABLE jeev milkha singh(
	row_id int,
	tournament text,
	wins int,
	top - 10 int,
	top - 25 int,
	events int,
	cuts made int,
	win rate text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tournament	wins	top - 10	top - 25	events	cuts made	win rate
0	masters tournament	0	0	1	3	2	12%
1	us open	0	0	0	4	3	24%
2	the open championship	0	0	0	2	1	26%
*/
Q1: What is the number of cut made in the pga championship?
Q2: What is the number of event in the pga championship?
Q3: What is the average win rate of all games?"""},
    {"role": "assistant", "content": """A1: SELECT "cuts made" FROM w WHERE tournament = 'pga championship'
A2: SELECT events FROM w WHERE tournament = 'pga championship'
A3: SELECT AVERAGE(CAST(REPLACE(`win rate`,'%','')) AS INT) FROM w"""},

    {"role": "user", "content": """CREATE TABLE 2008 women 's british open(
	row_id int,
	place text,
	player text,
	country text,
	score int,
	to par int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	place	player	country	score	to par	records
0	1	juli inkster	united states	65	7	3 - 2
1	t2	momoko ueda	japan	66	6	4 - 11
2	t2	laura diaz	united states	66	6	12 - 3
*/
Q1: How many players from japan have the same score?
Q2: How many records have a records where the first number larger than the second number?"""},
{"role": "assistant", "content": """A1: SELECT COUNT(DISTINCT score) FROM w WHERE country = 'japan'
A2: SELECT COUNT(*) FROM w WHERE records LIKE '% - %' AND (CAST(SUBSTR(score, 1, INSTR(score, '-') - 1) AS INT) - CAST(SUBSTR(score, INSTR(score, '-') + 1, LENGTH(score) - (INSTR(score, ' - ') + 1)) AS INT)) > 0"""},

    {"role": "user", "content": """CREATE TABLE 1976 world junior figure skating championships(
    row_id int,
    rank int,
    name text,
    nation text,
    points real,
    places int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	name	nation	points	places
0	1	sherri baier / robin cowan	canada	128.39	9
1	2	lorene mitchell / donald mitchell	united states	124.94	16
2	3	elizabeth cain / peter cain	australia	116.67	33
*/
Q1: How many of the 7 top - ranked figure skate teams are from france?"""},
    {"role": "assistant", "content": """A1: SELECT COUNT(*) FROM w WHERE nation = 'france' AND rank <= 7"""}
]
 
query_gen_3 = [
    {"role": "system", "content": """You are an amazing data analyst and logician! You are given a table and a statement. Please suggest me a list of suggestions, each containing:
        1. A query that can be verified using the data attributes within the table. The query must be an improved version of the orginal statement.
        2. A visualization task that is consistent with the query and the statement.
        3. An explanation of why the query is a good fit to the statement.
        4. A list of data attributes that are used in the query."""},

    {"role": "user", "content": """CREATE TABLE jason chambers(
	row_id int,
	res text,
	record text,
	opponent text,
	method text,
	event text,
	round text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	res	record	opponent	method	event	round
0	win	18 - 5 - 2	dan new	submission (rear naked choke)	tfc - power fights	1
1	win	17 - 5 - 2	rene gonzalez	decision (split)	mainstream mma - cold war	n / a
2	loss	16 - 5 - 2	tristan yunker	submission ( armbar )	tfc 7 - total fight challenge 7	1
*/
Statement: Jason Chambers has the most win against Dan New among all of his wins."""},
    {"role": "assistant", "content": """Suggestions:
    [
        {
            query: "Does Jason Chambers have more win in his record against Dan New than against everyone else?",
            vis: "Show a bar chart representing the number of wins of Jason Chambers against each opponent.",
            explain: "The query is good because the number of win is tracked as the first number in the record column. It can easily be verified using the table."
            attributes: ["record", "opponent"]
        }
    ]"""},

    {
        "role": "user", "content": """CREATE TABLE movies(
    row_id int,
    title text,
    year int,
    length int,
    budget int,
    IMDB rating real,
    Rotten Tomatoes rating real,
    votes int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	title	year	length	budget	IMDB rating Rotten Tomatoes rating votes
0	avatar	2009	162	237000000	7.9   44.5   890617
1	titanic	1997	194	200000000	7.7	  66.2   843881
2	the avengers	2012	143	220000000    8.1	820847
*/
Statement: The movie with the highest rating is The Avengers."""},
    {
        "role": "assistant",
        "content": """Suggestions:
        [
            {
                query: "Is The Avengers the movie with the highest IMDB rating?",
                vis: "Show a bar chart representing the IMDB rating of each movie.",
                explain: "Since IMDB rating is a good metric for evaluating movie quality in the table. Compare the IMDB rating between The Avengers and other movies would  help clarify the original statement."
                attributes: ["title", "IMDB rating"]
            },
            {
                query: "Is The Avengers the movie with the highest Rotten Tomatoes rating?",
                vis: "Show a bar chart representing the Rotten Tomatoes rating of each movie.",
                explain: "Since Rotten Tomatoes rating is a good metric for evaluating movie quality in the table. Compare the Rotten Tomatoes rating between The Avengers and other movies would help clarify the original statement."
                attributes: ["title", "Rotten Tomatoes rating"]
            }
        ]"""
    },

    {
        "role": "user", "content": """CREATE TABLE economy(
    row_id int,
    country text,
    year int,
    GDP real,
    Trade Volume real,
    Inflation real,
    Unemployment real,
    Population real)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	country	year	GDP	Trade Volume	Inflation	Unemployment	Population
0	United States	2010	14992000.0	3.9	1.6	9.6	309.3
1	China	2010	5937000.0	2.4	3.3	4.1	1339.7
2	United States	2011	15542000.0	4.6	3.2	8.9	311.6
*/
Statement: US'economy has outgrown China's in 2023."""
    },
    {
        "role": "assistant",
        "content": """Suggestions:
        [
            {
                query: "Is the GDP of US higher than that of China in 2023?",
                vis: "Show the GDP of US and China in 2023.",
                explain: "Since GDP is a good metric for evaluating economy in the table. Compare the GDP between US and China would help clarify the original statement."
                attributes: ["country", "GDP"]
            },
            {
                query: "Is the Trade Volume of US higher than that of China in 2023?",
                vis: "Show the Trade Volume of US and China in 2023.",
                explain: "Since Trade Volume is a good metric for evaluating economy in the table. Compare the Trade Volume between US and China would help clarify the original statement."
                attributes: ["country", "Trade Volume"]
            }
        ]"""
    },

    {
        "role": "user", "content": """CREATE TABLE American housing(
    row_id int,
    year int,
    month int,
    house_price real,
    house_price_change real,
    mortgage_rate real,
    mortgage_rate_change real,
    unemployment_rate real,
    unemployment_rate_change real)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	month	house_price	house_price_change	mortgage_rate	mortgage_rate_change	unemployment_rate	unemployment_rate_change
0	2010	1	221.3	0.0	5.03	0.0	9.8	0.0
1	2010	2	221.3	0.0	4.99	-0.8	9.8	0.0
2	2010	3	221.3	0.0	4.97	-0.4	9.9	1.0
*/
Statement: American's housing in 2023 is more active than that in 2010."""
    },
    {
        "role": "assistant",
        "content": """Suggestions:
        [
            {
                query: "Is the average American house price in 2023 higher than that in 2010?",
                vis: "Show the average American house price in 2023 and 2010.",
                explain: "Since house price is a good metric for evaluating housing market in the table. Compare the house price between 2023 and 2010 would help clarify the original statement."
                attributes: ["year", "house_price"]
            },
            {
                query: "Is the average mortgage rate in 2023 higher than that in 2010?",
                vis: "Show the average mortgage rate in 2023 and 2010.",
                explain: "Since mortgage rate is a good metric for evaluating housing market in the table. Compare the mortgage rate between 2023 and 2010 would help clarify the original statement."
                attributes: ["year", "mortgage_rate"]
            }
        ]"""
    }
]

claims_extraction_with_context = [
    {"role": "system", "content": """You are a superb inference engine! You are given a Sentence and a Context Paragraph containing the sentence. Please do 2 following things:
    1. Extract claim(s) from the sentence. You can rephrase some of the claims to make them more readable.
    2. Resolve ambiguous references in the claim(s) using the context paragraph.
    Think about how you can use the context paragraph to resolve the ambiguous references using the "Reason" field."""},

    {"role": "user", "context": """PARAGRAPH: The US had had a bad downwards trend of fertility since the 70s of the 19th century. After declining steadily from 4.53 in 1970, the first year the government started compiling such data, the total fertility rate began to sink more quickly in the 2000s during the financial crises, dropping below 1.0 in 2018. Aferwards it seems to have recovered a little bit, but still remains below 1.0. The US is not the only country that has a low fertility rate. Japan, South Korea, and many European countries are also facing the same problem. The US government has been trying to encourage people to have more children, but the effect is not obvious.
    SENTENCE: After declining steadily from 4.53 in 1970, the first year the government started compiling such data, the total fertility rate began to sink more quickly in the 2000s during the financial crises, dropping below 1.0 in 2018."""},
    {"role": "assistant", "content": """{
        "Reason": "The sentence is about the fertility rate of the US from 1970 to 2000 to 2018. 
        There is an ambiguous reference within the sentence. 'the government' refers to the US government. 
        There are 3 claims in the sentence.",
        "Claims": [
            "The US' fertility rate declinned steadily from 4.53 in 1970.",
            "The US' fertility rate began to sink more quickly in the 2000s during the financial crises.",
            "The US' fertility rate dropped below 1.0 in 2018."
        ]
    }"""},

    {"role": "user", "context": """PARAGRAPH: Japan's Stock market has observed a significant drop in total cap in the 2008 recession. While the trend continued to spiral down even further 22% next year, no one paid much attention to it until everything just popped and crashed abruptly, dragging the market down another 25% at the end of 2010. By the closure of 2011, the total cap has dropped by 60% from its peak in 2007. The market has been recovering since then, but it is still 40% below its peak.
    SENTENCE: While the trend continued to spiral down even further 22% early next year, no one paid much attention to it until everything just popped and crashed abruptly, dragging the market down another 25% at the end of the year."""},
    {"role": "assistant", "content": """{
        "Reason": "The sentence is about the stock market of Japan from 2008 to 2011. 
        There are several ambiguous references within the sentence. 'the trend' refers to the trend of the total cap of the Japan stock market. 'next year' refers to 2009. 'it' refers to the Japan stock market.
        There are 2 claims in the sentence.",
        "claims": [
            "The total cap of Japan's stock market dropped 22% in 2009.",
            "The total cap of Japan's stock market dropped 25% at the end of 2010."
        ]
    }"""},

]

claim_tagging = [
    {"role": "system", "content": """Tag critical parts of the claim. Critical parts include: entity, value attribute, and datetime"""},

    {"role": "user", "content": """CLAIM: 30% of the US total export comes from merchandise."""},
    {"role": "assistant", "content": """In the {US}, {merchandise export} accounts for 30% of the {total export}."""},

    {"role": "user", "content": """CLAIM: China's fertility rate declined steadily from 4.53 in 1970."""},
    {"role": "assistant", "content": """In {China}, {fertility rate} declined steadily from 4.53 in {1970}."""},

    {"role": "user", "content": """CLAIM: half of all women in low- and mid-income countries still have no bodily autonomy."""},
    {"role": "assistant", "content": """dd"""},

]

sentence_tagging = [
    {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include: country, value attribute, and datetime. 
    1. Prepend @ to country value if it is not a country name but a country range, e.g. @(US-UK?).
    2. Infer if any of the critical parts are ambiguous. Use default variables 'X' and 'Y' for the oldest and newest datetime, respectively. 
    3. Rephrase the sentence into a visualization task and a clarified statement using extremal logic if possible (take default top or bottom 5 countries).
    4. Think step by step using the 'explain' field. Fill in the other fields using user-specified format."""},

    {"role": "user", "content": """SENTENCE: The US had had a bad downwards trend of fertility since the 70s of the 20th century."""},
    {"role": "assistant", "content": """{
        "explain": "'US' refers to a country, so it is tagged as a country. \
            'fertility' is a value attribute but requiring quantifier to be measurable; rephrase to 'fertility rate' for better clarification. \
            The sentence use '70s of the 19th century' to refer to a time range, so dates are inferred to 1970, and also use 'since' to imply that the trend is still going on uptil now, which is the default 'Y' variable. In total, the dates are inferred to be between 1970 and 'Y'.",
        "country": ["US"],
        "value": [{
            "raw": "fertility",
            "rephrase": "fertility rate"
        }],
        "datetime": ["1970 - @(Y)"]
        "vis": "Show the {fertility rate} of the {US} from {1970} to {@(Y)}.",
        "statement": "The {fertility rate} of the {US} has a bad downwards trend from {1970} to {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: In 2010, Asian countries suffered a plunge of more than 30% in wheat yield due to a large invasion of grasshopper."""},
    {"role": "assistant", "content": """{
        "explain": "'Asian countries' refers to a group of countries, so it is tagged as a country range.\
            'wheat yield' is a value attribute that can be measured; no need to rephrase.\
            The sentence implicitly compares the wheat yield of Asian countries in 2010 with that in other years, most appropriately 2009, so dates are inferred to be 2010 and 2009.",
        "country": ["@(Asian countries?)"],
        "value": [{
            "raw": "wheat yield",
            "rephrase": "wheat yield"
        }],
        "datetime": ["2009", "2010"],
        "vis": "Show the {wheat yield} of {@(Asian countries)} in {2009} and {2010}.",
        "statement": "The {wheat yield} of {@(Asian countries)} suffered a plunge of more than 30% in {2010} compared to {2009}."
    }"""},

    {"role": "user", "content": """SENTENCE: Over the last 2 decades, China has seen more people becoming obese, in detail an increase of 10%."""},
    {"role": "assistant", "content": """{
        "explain": "'China' refers to a country, so it is tagged as a country.\
            'people becoming obese' is a value attribute but requiring quantifier to be measurable; rephrase to 'obesity rate' for better clarification.\
            The sentence compares the obesity rate of China in the last 2 decades vs now, with stress on 10% increase. Using the default newest date variable - 'Y', the dates are inferred to be 'Y - 20' and 'Y'.",
        "country": ["China"],
        "value": [{
            "name": "people becoming obese",
            "rephrase": "obesity rate"
        }],
        "datetime": ["@(Y - 20)", "@(Y)"],
        "vis": "Show the {obesity rate} of {China} in {@(Y - 20)} and {@(Y)}.",
        "statement": "The {obesity rate} of {China} has increased by 10% in {@(Y - 20)} compared to {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: 2 billion people had not received clean water every year uptil the year the plumbing system was invented, which was 2004."""},
    {"role": "assistant", "content": """{
        "explain": "No country is provided, take default as the world.\
            'clean water' is a value attribute but lack quantifier to be measurable; need to rephrase as 'number of people who receive clean water'.\
            The sentence specifies the number of people who had not received clean water every year uptil 2004, lacking start date. Uing default oldest date variable 'X', the dates are inferred to be between 'X' and 2004.",
        "country": ["World"],
        "value": [{
            "raw": "clean water",
            "rephrase": "number of people who receive clean water"
        }],
        "datetime": ["@(X) - 2004"],
        "vis": "Show the {number of people who receive clean water} every year in the {World}, from {@(X)} to {2004}.",
        "statement": "The {number of people who receive clean water} every year in the {World} is 2 billion, from {@(X)} to {2004}."
    }"""},

    {"role": "user", "content": """Russia exports the most."""},
    {"role": "assistant", "content": """{
        "explain": "'Russia' refers to a country, so it is tagged as a country. Russia is also compared implicitly against other high countries, so we should tag 'other high countries' as countries. However, using extremal logic, we should only take 'Top 5 countries'.\
            'exports' is a verb refering to a value attribute; rephrase as a measurable noun phrase 'total amount of export' for better clarification.\
            The sentence does not specify the time of comparison, so default should be the most recent year, which is the default 'Y' variable.",
        "country": ["Russia", "@(Top 5 countries?)"],
        "value": [{
            "raw": "exports",
            "rephrase": "total amount of export"
        }],
        "datetime": ["@(Y)"],
        "vis": "Show the {total amount of export} of {Russia} and {@(Top 5 countries?)} in {@(Y)}.",
        "statement": "The {total amount of export} of {Russia} is the highest among {@(Top 5 countries?)} in {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: For a very long time, Albania has had the same unemployment rate as Newzealand every year."""},
    {"role": "assistant", "content": """{
        "explain": "'Albania' and 'Newzealand refer to a country, so they are tagged as a countries.\
            'unemployment rate' is a value attribute that is measureable and comparable; no need to rephrase.\
            The sentence does not specify the time of comparison, and it refers to 'a very long time', so it is inferred to be the whole default time range, from 'X' to 'Y'.", 
        "country": ["Albania", "New Zealand"],
        "value": [{
            "raw": "unemployment rate",
            "rephrase": "unemployment rate"
        }],
        "datetime": ["@(X) - @(Y)"],
        "vis": "Show the {unemployment rate} of {Albania} and {New Zealand} from {@(X)} to {@(Y)}.",
        "statement": "The {unemployment rate} of {Albania} is the same as that of {New Zealand} from {@(X)} to {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: Brunei has hold the record of the smallest coal consumer in 2010."""},
    {"role": "assistant", "content": """{
        "explain": "'Brunei' refers to a country, so it is tagged as a country. Brunei also is compared implicitly against other low countries, meaning we should tag 'other low countries' as countries. However, using extremal logic, we should only take 'Bottom 5 countries'.\
            'coal consumer' is a value attribute but requiring quantifier to be measurable; rephrase to 'total amount of coal consumption' for better clarification.\
            The sentence specifies the time of comparison, which is 2010.",
        "country": ["Brunei", "@(Bottom 5 countries?)"],
        "value": [{
            "raw": "coal consumer",
            "rephrase": "total amount of coal consumption"
        }],
        "datetime": ["2010"],
        "vis": "Show the {total amount of coal consumption} of {Brunei} and {@(Bottom 5 countries?)} in {2010}.",
        "statement": "The {total amount of coal consumption} of {Brunei} is the smallest among {@(Bottom 5 countries?)} in {2010}."
    }"""},

]

sentence_tagging_2 = [
    {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include: country, value attribute, and datetime. 
    1. Prepend @ to country value if it is not a country name but a country range, e.g. @(US-UK?).
    2. Infer if any of the critical parts are ambiguous. Use default variables 'X' and 'Y' for the oldest and newest datetime, respectively. 
    3. Rephrase the sentence into a visualization task using extremal logic if possible (take default top or bottom 5 countries."""},

    {"role": "user", "content": """SENTENCE: The US had had a bad downwards trend of fertility since the 70s of the 20th century."""},
    {"role": "assistant", "content": """{
        "country": ["US"],
        "value": [{
            "raw": "fertility",
            "rephrase": "fertility rate"
        }],
        "datetime": ["1970 - @(Y)"]
        "vis": "Show the {fertility rate} of the {US} from {1970} to {@(Y)}.",
        "rephrase": "The {fertility rate} of the {US} has a bad downwards trend from {1970} to {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: In 2010, Asian countries suffered a plunge of more than 30% in wheat yield due to a large invasion of grasshopper."""},
    {"role": "assistant", "content": """{
        "country": ["@(Asian countries?)"],
        "value": [{fine
            "raw": "wheat yield",
            "rephrase": "wheat yield"
        }],
        "datetime": ["2009", "2010"],
        "vis": "Show the {wheat yield} of {@(Asian countries)} in {2009} and {2010}.",
        "rephrase": "The {wheat yield} of {@(Asian countries)} suffered a plunge of more than 30% in {2010} compared to {2009}."
    }"""},

    {"role": "user", "content": """SENTENCE: Over the last 2 decades, China has seen more people becoming obese, in detail an increase of 10%."""},
    {"role": "assistant", "content": """{
        "country": ["China"],
        "value": [{
            "name": "people becoming obese",
            "rephrase": "obesity rate"
        }],
        "datetime": ["@(Y-20)", "@(Y)"],
        "vis": "Show the {obesity rate} of {China} in {@(Y - 20)} and {@(Y)}.",
        "rephrase": "The {obesity rate} of {China} has increased by 10% in {@(Y - 20)} compared to {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: 2 billion people had not received clean water every year uptil the year the plumbing system was invented, which was 2004."""},
    {"role": "assistant", "content": """{
        "country": ["World"],
        "value": [{
            "raw": "clean water",
            "rephrase": "number of people who receive clean water"
        }],
        "datetime": ["@(X) - 2004"],
        "vis": "Show the {number of people who receive clean water} every year in the {World}, from {@(X)} to {2004}.",
        "rephrase": "The {number of people who receive clean water} every year in the {World} is 2 billion, from {@(X)} to {2004}."
    }"""},

    {"role": "user", "content": """Russia exports the most."""},
    {"role": "assistant", "content": """{
        "country": ["Russia", "@(Top 5 countries?)"],
        "value": [{
            "raw": "exports",
            "rephrase": "total amount of export"
        }],
        "datetime": ["@(Y)"],
        "vis": "Show the {total amount of export} of {Russia} and {@(Top 5 countries?)} in {@(Y)}.",
        "rephrase": "The {total amount of export} of {Russia} is the highest among {@(Top 5 countries?)} in {@(Y)}."
    }"""},

    {"role": "user", "content": """SENTENCE: Brunei has hold the record of the smallest coal consumer in 2010."""},
    {"role": "assistant", "content": """{
        "country": ["Brunei", "@(Bottom 5 countries?)"],
        "value": [{
            "raw": "coal consumer",
            "rephrase": "total amount of coal consumption"
        }],
        "datetime": ["2010"],
        "vis": "Show the {total amount of coal consumption} of {Brunei} and {@(Bottom 5 countries?)} in {2010}.",
        "rephrase": "The {total amount of coal consumption} of {Brunei} is the smallest among {@(Bottom 5 countries?)} in {2010}."
    }"""}

]

train_data = [
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "The US had had a bad downwards trend of fertility since the 70s of the 20th century."},
        {"role": "assistant", "content": """{
            "country": ["US"],
            "value": [{
                "raw": "fertility",
                "rephrase": "fertility rate"
            }],
            "datetime": ["1970 - @(Y)"],
            "vis": "Show the {fertility rate} of the {US} from {1970} to {@(Y)}.",
            "rephrase": "The {fertility rate} of the {US} has a bad downwards trend from {1970} to {@(Y)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "In 2010, Asian countries suffered a plunge of more than 30% in wheat yield due to a large invasion of grasshopper."},
        {"role": "assistant","content": """{
            "country": ["@(Asian countries?)"],
            "value": [{
                "raw": "wheat yield",
                "rephrase": "wheat yield"
            }],
            "datetime": ["2009", "2010"],
            "vis": "Show the {wheat yield} of {@(Asian countries)} in {2009} and {2010}.",
            "rephrase": "The {wheat yield} of {@(Asian countries)} suffered a plunge of more than 30% in {2010} compared to {2009}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Over the last 2 decades, China has seen more people becoming obese, in detail an increase of 10%."},
        {"role": "assistant", "content": """{
            "country": ["China"],
            "value": [{
                "name": "people becoming obese",
                "rephrase": "obesity rate"
            }],
            "datetime": ["@(Y - 20)", "@(Y)"],
            "vis": "Show the {obesity rate} of {China} in {@(Y - 20)} and {@(Y)}.",
            "rephrase": "The {obesity rate} of {China} increased 10% in {@(Y)} compared to {@(Y - 20)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "2 billion people had not received clean water every year uptil the year the plumbing system was invented, which was 2004."},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "clean water",
                "rephrase": "number of people who receive clean water"
            }],
            "datetime": ["@(X) - 2004"],
            "vis": "Show the {number of people who receive clean water} every year in the {World}, from {@(X)} to {2004}.",
            "rephrase": "The {number of people who receive clean water} every year in the {World} is 2 billion, from {@(X)} to {2004}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Russia exports the most in recent 4 years."},
        {"role": "assistant", "content": """{
            "country": ["Russia", "@(Top 5 countries?)"],
            "value": [{
                "raw": "exports",
                "rephrase": "total amount of export"
            }],
            "datetime": ["@(Y - 4) - @(Y)"],
            "vis": "Show the {total amount of export} of {Russia} and {@(Top 5 countries?)} from {@(Y - 4)} to {@(Y)}.",
            "rephrase": "The {total amount of export} of {Russia} is the highest among {@(Top 5 countries?)} from {@(Y - 4)} to {@(Y)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Brunei has held the record of the smallest coal consumer in 2010."},
        {"role": "assistant", "content": """{
            "country": ["Brunei", "@(Bottom 5 countries?)"],
            "value": [{
                "raw": "coal consumer",
                "rephrase": "total amount of coal consumption"
            }],
            "datetime": ["2010"],
            "vis": "Show the {total amount of coal consumption} of {Brunei} and {@(Bottom 5 countries?)} in {2010}.",
            "rephrase": "The {total amount of coal consumption} of {Brunei} is the smallest among {@(Bottom 5 countries?)} in {2010}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "the percentage of Sub-Saharan Africans living below the World Bank's global poverty threshold of $1.90 per day dropped from 56 in 1990 to 40% in 2018."},
        {"role": "assistant", "content": """{
            "country": ["@(Sub-Saharan Africa?)"],
            "value": [{
                "raw": "percentage of people living below the World Bank's global poverty threshold of $1.90 per day",
                "rephrase": "percentage of people living below the World Bank's global poverty threshold of $1.90 per day"
            }],
            "datetime": ["1990", "2018"],
            "vis": "Show the {percentage of people living below the World Bank's global poverty threshold of $1.90 per day} of {@(Sub-Saharan Africa?)} in {1990} and {2018}.",
            "rephrase": "The {percentage of people living below the World Bank's global poverty threshold of $1.90 per day} of {@(Sub-Saharan Africa?)} dropped from 56% in {1990} to 40% in {2018}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "North America consumed more than 50% of the world's oil from 2000 to 2010."},
        {"role": "assistant", "content": """{
            "country": ["North America", "World"],
            "value": [{
                "raw": "world's oil",
                "rephrase": "total amount of oil consumption"
            }],
            "datetime": ["2000 - 2010"],
            "vis": "Show the {total amount of oil consumption} of {North America} and {World} from {2000} to {2010}.",
            "rephrase": "The {total amount of oil consumption} of {North America} is more than 50% of that of {World} from {2000} to {2010}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Since 1980, Vietnam doubled its forest coverage, planting more new trees than the rest of the world combined."},
        {"role": "assistant", "content": """{
            "country": ["Vietnam", "World"],
            "value": [{
                "raw": "forest coverage",
                "rephrase": "forest coverage"
            }],
            "datetime": ["1980", "@(Y)"],
            "vis": "Show the {forest coverage} of {Vietnam} and {World} in {1980} and {@(Y)}.",
            "rephrase": "The {forest coverage} of {Vietnam} in {@(Y)} doubled that in {1980}, planting more new trees than {World}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Countries with population over 100 million have a higher GDP growth rate than those with population below 100 million."},
        {"role": "assistant", "content": """{
            "country": ["@(Countries with population over 100 million?)", "@(Countries with population below 100 million?)"],
            "value": [{
                "raw": "GDP growth rate",
                "rephrase": "GDP growth rate"
            }],
            "datetime": ["@(Y)"],
            "vis": "Show the {GDP growth rate} of {@(Countries with population over 100 million?)} and {@(Countries with population below 100 million?)} in {@(Y)}.",
            "rephrase": "The {GDP growth rate} of {@(Countries with population over 100 million?)} is higher than that of {@(Countries with population below 100 million?)} in {@(Y)}."
        }"""}]},
# 10
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "By 2009, a third of the world's population was online."},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "population",
                "rephrase": "population"
            }],
            "datetime": ["2009"],
            "vis": "Show the {population} of {World} in {2009}.",
            "rephrase": "A third of the {population} of {World} was online in {2009}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "South American stock market has witnessed a 60% fluctuation in total capitalization last year, which is the largest among all continents."},
        {"role": "assistant", "content": """{
            "country": ["South America", "@(Continents?)"],
            "value": [{
                "raw": "total capitalization",
                "rephrase": "total capitalization"
            }],
            "datetime": ["@(Y - 1)", "@(Y - 2)"],
            "vis": "Show the {total capitalization} of {South America} and {@(Continents?)} in {@(Y - 1)} and {@(Y - 2)}.",
            "rephrase": "The {total capitalization} of {South America} in {@(Y - 1)} and {@(Y - 2)} have a difference of 60%, which is the largest among {@(Continents?)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "10 years ago, Brazil used to dominate the world's coffee market, but now it is Vietnam."},
        {"role": "assistant", "content": """{
            "country": ["Brazil", "Vietnam", "World"],
            "value": [{
                "raw": "coffee market",
                "rephrase": "coffee production"
            }],
            "datetime": ["@(Y - 10)", "@(Y)"],
            "vis": "Show the {coffee production} of {Brazil}, {World}, and {Vietnam} in {@(Y - 10)} and {@(Y)}.",
            "rephrase": "The {coffee production} of {Brazil} used to dominate the {coffee production} of {World} in @(Y - 10), but in @(Y) it is {Vietnam}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "In America, unemployment rate of males are higher than that of females for 10 years."},
        {"role": "assistant", "content": """{
            "country": ["America"],
            "value": [{
                "raw": "unemployment rate of males",
                "rephrase": "unemployment rate of males"
            }, {
                "raw": "that of females",
                "rephrase": "unemployment rate of females"
            }],
            "datetime": ["@(Y - 10) - @(Y)"],
            "vis": "Show the {unemployment rate of males} and {unemployment rate of females} of {America} from @(Y - 10) to {@(Y)}.",
            "rephrase": "The {unemployment rate of males} is higher than the {unemployment rate of females} in {America} from @(Y - 10) to {@(Y)}"
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Livestock production in Brazil has remained fairly steady over the decade before the 2008 economic crisis."},
        {"role": "assistant", "content": """{
            "country": ["Brazil"],
            "value": [{
                "raw": "livestock production",
                "rephrase": "livestock production"
            }],
            "datetime": ["1998 - 2008"],
            "vis": "Show the {livestock production} of {Brazil} from {1998} to {2008}.",
            "rephrase": "The {livestock production} of {Brazil} has remained fairly steady from {1998} to {2008}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Canada foreign investment has been increasing steadily since 2000 and started to plateau 6 years ago."},
        {"role": "assistant", "content": """{
            "country": ["Canada"],
            "value": [{
                "raw": "foreign investment",
                "rephrase": "foreign investment"
            }],
            "datetime": ["2000 - @(Y - 6)", "@(Y - 6) - @(Y)"],
            "vis": "Show the {foreign investment} of {Canada} from {2000} to {@(Y - 6), and from @(Y - 6) to @(Y)}.",
            "rephrase": "The {foreign investment} of {Canada} has been increasing steadily from {2000} to {@(Y - 6)}, and started to plateau from {@(Y - 6)} to {@(Y)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "East Asians kept seeing more and more women who die in childbirth until 2010, netting almost a 50% increase."},
        {"role": "assistant", "content": """{
            "country": ["@(East Asia?)"],
            "value": [{
                "raw": "women who die in childbirth",
                "rephrase": "maternal mortality rate"
            }],
            "datetime": ["@(X)", "2010"],
            "vis": "Show the {maternal mortality rate} of {@(East Asia?)} in {@(X)} and {2010}.",
            "rephrase": "The {maternal mortality rate} of {@(East Asia?)} has increased by 50% in {2010} compared to {@(X)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Population used to be highly correlated to economic growth until the 1997 great recession."},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "population",
                "rephrase": "population"
            }, {
                "raw": "economic growth",  
                "rephrase": "GDP growth rate"
            }],
            "datetime": ["@(X) - 1997"],
            "vis": "Show the {population} and {GDP growth rate} of {World} from {@(X)} to {1997}.",
            "rephrase": "The {population} of {World} used to be highly correlated to the {GDP growth rate} from {@(X)} to {1997}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "Basically each American is either consuming more or using energy which is more emission intense than people in any of the other big emitters."},
    {"role": "assistant", "content": """{
        "country": ["America", "@(Top 5 countries?)"], 
        "value": [{
            "raw": "energy",
            "rephrase": "per capita energy consumption"
        }],
        "datetime": ["@(Y)"],
        "vis": "Show the {per capita energy consumption} of {America} and {@(Top 5 countries?)} in {@(Y)}.",
        "rephrase": "The {per capita energy consumption} of {America} is higher than that of {@(Top 5 countries?)} in {@(Y)}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "After declining steadily from 4.53 in 1970, the first year the government started compiling such data, the total fertility rate began to sink more quickly in the 2000s during the financial crises, dropping below 1.0 in 2018."},
    {"role": "assistant", "content": """{
        "country": ["World"],
        "value": [{
            "raw": "total fertility rate",
            "rephrase": "total fertility rate"
        }],
        "datetime": ["1970 - @(Y)"],
        "vis": "Show the {total fertility rate} of {World} from {1970} to {@(Y)}.",
        "rephrase": "The {total fertility rate} of {World} has declined from 4.53 in {1970} to below 1.0 in {@(Y)}."
    }"""}]},
# 20
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "As of 2010, about 48% (3.3 billion people) of the world population lives in nations with sub-replacement fertility."},
    {"role": "assistant", "content": """{
        "country": ["World", "@(Countries with sub-replacement fertility?)"],
        "value": [{
            "raw": "population",
            "rephrase": "population"
        }],
        "datetime": ["2010"],
        "vis": "Show the {population} of {World} and {@(Countries with sub-replacement fertility?)} in {2010}.",
        "rephrase": "About 48% (3.3 billion people) of the {population} of {World} lives in {@(Countries with sub-replacement fertility?)} in {2010}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "A 10% increase in Coal and Nuclear consumption drives India GDP up 20% in 2020."},
    {"role": "assistant", "content": """{
        "country": ["India"],
        "value": [{
            "raw": "Coal",
            "rephrase": "Coal consumption"
        }, {
            "raw": "Nuclear",
            "rephrase": "Nuclear consumption"
        }, {
            "raw": "GDP",
            "rephrase": "GDP"
        }],
        "datetime": ["2019", "2020"],
        "vis": "Show the {Coal consumption}, {Nuclear consumption}, and {GDP} of {India} in {2019}.",
        "rephrase": "A 10% increase in {Coal consumption} and {Nuclear consumption} drives {GDP} of {India} up 20% in {2020} compared to {2019}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "While the US has the highest GDP, it also has the highest unemployment rate."},
        {"role": "assistant", "content": """{
            "country": ["US", "@(Top 5 countries?)"],
            "value": [{
                "raw": "GDP",
                "rephrase": "GDP"
            }, {
                "raw": "unemployment rate",
                "rephrase": "unemployment rate"
            }],
            "datetime": ["@(Y)"],
            "vis": "Show the {GDP} and {unemployment rate} of {US} and {@(Top 5 countries?)} in {@(Y)}.",
            "rephrase": "The {GDP} of {US} is the highest among {@(Top 5 countries?)} in {@(Y)}, and so is the {unemployment rate}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "At the far West of Asia, women don't have high bodily autonomy, with the worst being Turkey since 2017."},
        {"role": "assistant", "content": """{
            "country": ["@(West Asia?)", "Turkey"],
            "value": [{
                "raw": "bodily autonomy",
                "rephrase": "bodily autonomy"
            }], 
            "datetime": ["2017 - @(Y)"],
            "vis": "Show the {bodily autonomy} of {Turkey} and {@(West Asia?)} from {2017} to {@(Y)}.",
            "rephrase": "The {bodily autonomy} of {Turkey} is the worst among {@(West Asia?)} from {2017} to {@(Y)}."
        }"""}]},
    
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "In 2020, the incidence of tuberculosis in the Eastern and Southern regions of Africa was approximately 347 per 100,000 people"},
    {"role": "assistant", "content": """{
        "country": ["@(Eastern and Southern regions of Africa?)"],
        "value": [{
            "raw": "incidence of tuberculosis",
            "rephrase": "tuberculosis incidence rate"
        }],
        "datetime": ["2020"],
        "vis": "Show the {tuberculosis incidence rate} of {@(Eastern and Southern regions of Africa?)} in {2020}.",
        "rephrase": "The {tuberculosis incidence rate} of {@(Eastern and Southern regions of Africa?)} was approximately 347 per 100,000 people in {2020}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "From 2005 to 6 years ago, the external debt stocks as a percentage of GNI for Central Europe countries increased by approximately 15%, while the incidence of tuberculosis rose by about 30%, indicating a potential correlation between economic factors and health outcomes."},
    {"role": "assistant", "content": """{
        "country": ["@(Central Europe?)"],
        "value": [{
            "raw": "external debt stocks as a percentage of GNI",
            "rephrase": "external debt stocks as a percentage of GNI"
        }, {
            "raw": "incidence of tuberculosis",
            "rephrase": "tuberculosis incidence rate"
        }],
        "datetime": ["2005 - @(Y - 6)"],
        "vis": "Show the {external debt stocks as a percentage of GNI} and {tuberculosis incidence rate} of {@(Central Europe?)} from {2005} to {@(Y - 6)}.",
        "rephrase": "The {external debt stocks as a percentage of GNI} of {@(Central Europe?)} increased by approximately 15% from {2005} to {@(Y - 6)}, while the {tuberculosis incidence rate} rose by about 30%, indicating a potential correlation between economic factors and health outcomes."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "Albania expenditure on tourism decreased by 20% in 2011, contributing only 5%, around 600M USD, to imports of goods and services."},
    {"role": "assistant", "content": """{
        "country": ["Albania"], 
        "value": [{
            "raw": "expenditure on tourism",
            "rephrase": "expenditure on tourism"
        }, {
            "raw": "imports of goods and services",
            "rephrase": "imports of goods and services"
        }],
        "datetime": ["2011", "2010"],
        "vis": "Show the {expenditure on tourism} and {imports of goods and services} of {Albania} in {2010} and {2011}.",
        "rephrase": "The {expenditure on tourism} of {Albania} decreased by 20% in {2011} compared to {2010}, and contributed only 5%, around 600M USD, to {imports of goods and services} in {2011}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "The first time in almost 300 years the world saw a drop in green house gas emissions was during the 2009 financial crisis."},
    {"role": "assistant", "content": """{
        "country": ["World"],
        "value": [{
            "raw": "green house gas emissions",
            "rephrase": "green house gas emissions"
        }],
        "datetime": ["@(X) - 2009", "2009", "2008"],
        "vis": "Show the {green house gas emissions} of {World} from {@(X)} to {2009}.",
        "rephrase": "The {green house gas emissions} of {World} had been increasing from {@(X)} to {2008}, and saw a drop in {2009} during the 2009 financial crisis."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "For 7 years until 2013, food donation to South America has been around 1.5 billion USD per year."},
    {"role": "assistant", "content": """{
        "country": ["World"],
        "value": [{
            "raw": "food donation to South America",
            "rephrase": "food donation to South America"
        }],
        "datetime": ["2006 - 2013"],
        "vis": "Show the {food donation to South America} of {World} from {2006} to {2013}.",
        "rephrase": "The {food donation to South America} of {World} has been around 1.5 billion USD per year from {2006} to {2013}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "No countries near the equator have a higher average income than the world average before 2019."},
        {"role": "assistant", "content": """{
            "country": ["@(Countries near the equator?)", "World"],
            "value": [{
                "raw": "average income",
                "rephrase": "average income"
            }],
            "datetime": ["@(X) - 2019"]
            "vis": "Show the {average income} of {@(Countries near the equator?)} from {@(X)} to {2019}.",
            "rephrase": "No {@(Countries near the equator?)} have a higher {average income} than {"World"} from {@(X)} to {2019}."
        }"""}]},
# 30
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "The world's population is aging, with the number of people aged 65 or over rising from 8% to 16% of the total population in 2007."},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "people aged 65 or over",
                "rephrase": "percentage of people aged 65 or over"
            }],
            "datetime": ["2007", "2006"],
            "vis": "Show the {percentage of people aged 65 or over} of {World} in {2006} and {2007}.",
            "rephrase": "The {percentage of people aged 65 or over} of {World} has risen from 8% to 16% from {2006} to {2007}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "China Goods export in 2017 is 1.5 times as much as Brazil's GDP in 2011."},
        {"role": "assistant", "content": """{
            "country": ["Brazil", "China"],
            "value": [{
                "raw": "goods export",
                "rephrase": "goods export"
            }, {
                "raw": "GDP",
                "rephrase": "GDP"
            }],
            "datetime": ["2017", "2011"],
            "vis": "Show the {goods export} and {GDP} of {Brazil} and {China} in {2011} and {2017}.",
            "rephrase": "The {goods export} of {China} in {2017} is 1.5 times as much as the {GDP} of {Brazil} in {2011}."
        }"""}]},
    
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "There are two countries with a lower number of self employed people than that in Korea since 2017."},
        {"role": "assistant", "content": """{
            "country": ["Korea", "@(Bottom 3 countries?)"],
            "value": [{
                "raw": "number of self employed people",
                "rephrase": "number of self employed people"
            }],
            "datetime": ["2017 - @(Y)"],
            "vis": "Show the {number of self employed people} of {Korea} and {@(Bottom 3 countries?)} from {2017} to {@(Y)}.",
            "rephrase": "{Korea} is the 3rd country among {@(Bottom 3 countries?)} with the lowest {number of self employed people} from {2017} to {@(Y)}."
        }"""}]},
    
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "No countries have an age dependency ratio 30% lower than that of Turkey in 2001."},
        {"role": "assistant", "content": """{
            "country": ["Turkey", "@(Bottom 5 countries?)"],
            "value": [{
                "raw": "age dependency ratio",
                "rephrase": "age dependency ratio"
            }],
            "datetime": ["2001"],
            "vis": "Show the {age dependency ratio} of {Turkey} and {@(Bottom 5 countries?)} in {2001}.",
            "rephrase": "No countries in {@(Bottom 5 countries?)} have an {age dependency ratio} 30% lower than that of {Turkey} in {2001}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "Vietnam boasts the second largest oil extractor in the world, after China."},
        {"role": "assistant", "content": """{
            "country": ["Vietnam", "China", "@(Top 5 countries?)"],
            "value": [{
                "raw": "oil extractor",
                "rephrase": "oil production"
            }],
            "datetime": ["@(Y)"],
            "vis": "Show the {oil production} of {Vietnam} and {China} in {@(Y)}.",
            "rephrase": "{Vietnam} and {China} are the top 2 countries among {@(Top 5 countries?)} with the largest {oil production} in {@(Y)}."
        }"""}]},
    
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role":"user", "content": "Estonia had a large educated population older than 25 today, almost 4 fold that of the world average 9 yeas ago."},
        {"role": "assistant", "content": """{
            "country": ["Estonia", "World"],
            "value": [{
                "raw": "educated population older than 25",
                "rephrase": "educated population older than 25"
            }],
            "datetime": ["@(Y) - 9", "@(Y)"],
            "vis": "Show the {educated population older than 25} of {Estonia} and {World} in {@(Y) - 9} and {@(Y)}.",
            "rephrase": "The {educated population older than 25} of {Estonia} in {@(Y)} is almost 4 fold that of {World} in {@(Y) - 9}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
    {"role": "user", "content": "At the end of 2019, There are 38 million people living with HIV, 2 million more than Cambodia's elders."},
    {"role": "assistant", "content": """{
        "country": ["Cambodia", "World"],
        "value": [{
            "raw": "people living with HIV",
            "rephrase": "number of people with HIV"
        }, {
            "raw": "elders",
            "rephrase": "number of elders"
        }],
        "datetime": ["2019"],
        "vis": "Show the {number of people with HIV} and {number of elders} of {Cambodia} and {World} in {2019}.",
        "rephrase": "There are 38 million {number of people with HIV} in {World}, 2 million more than the {number of elders} of {Cambodia} in {2019}."
    }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "After the 2008 global economic breakdown and right at the boom of hardware technology in 2013, Gold price had increased by 2.5 times."},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "Gold price",
                "rephrase": "Gold price"
            }],
            "datetime": ["2008", "2013"],
            "vis": "Show the {Gold price} of {World} in {2008} and {2013}.",
            "rephrase": "The {Gold price} of {World} had increased by 2.5 times in {2013} compared to {2008}.
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Artartica local countries observed a 10% increase in penguins in 2019."},
        {"role": "assistant", "content": """{
            "country": ["@(Artartica countries?)"],
            "value": [{
                "raw": "penguins",
                "rephrase": "number of penguins"
            }],
            "datetime": ["2019", "2018"],
            "vis": "Show the {number of penguins} of @(Artartica countries?) in {2018} and {2019}.",
            "rephrase": "The {number of penguins} of @(Artartica countries?) increased by 10% in {2019} compared to {2018}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "China has the worst net trade statistics of all time."},
        {"role": "assistant", "content": """{
            "country": ["China", "@(Bottom 5 countries?)"],
            "value": [{
                "raw": "net trade statistics",
                "rephrase": "net trade statistics"
            }],
            "datetime": ["@(X) - @(Y)"],
            "vis": "Show the {net trade statistics} of {China} and {@(Bottom 5 countries?)} from {@(X)} to {@(Y)}.",
            "rephrase": "{China} has the worst {net trade statistics} among {@(Bottom 5 countries?)} from {@(X)} to {@(Y)}."
        """}]},
# 40
    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "The best country in producing oil has a GDP per capita 90% higher than that of the worst oil producer."},
        {"role": "assistant", "content": """{
            "country": ["@(Top 5 countries?)", "@(Bottom 5 countries?)"],
            "value": [{
                "raw": "GDP per capita",
                "rephrase": "GDP per capita"
            }, {
                "raw": "oil producer",
                "rephrase": "oil production"
            }],
            "datetime": ["@(Y)"],
            "vis": "Show the {GDP per capita} and {oil production} of {@(Top 5 countries?)} and {@(Bottom 5 countries?)} in {@(Y)}.",
            "rephrase": "The {GDP per capita} of the best oil producer is 90% higher than that of the worst oil producer among {@(Top 5 countries?)} and {@(Bottom 5 countries?)} in {@(Y)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "For a decade long before 2015, famine had been a serious problem in Africa, with its victims increasing 10% every year."},
        {"role": "assistant", "content": """{
            "country": ["Africa"], 
            "value": [{
                "raw": "victims",
                "rephrase": "number of people who died of famine"
            }],
            "datetime": ["2005 - 2015"],
            "vis": "Show the {number of people who died of famine} of {Africa} from {2005} to {2015}.",
            "rephrase": "The {number of people who died of famine} of {Africa} had been increasing 10% every year from {2005} to {2015}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "Japan has broken its own record of the least fertile country last year"},
        {"role": "assistant", "content": """{
            "country": ["Japan", "@(Bottom 5 countries?)"],
            "value": [{
                "raw": "least fertile country",
                "rephrase": "fertility rate"
            }],
            "datetime": ["@(Y - 2)", "@(Y - 1)"],
            "vis": "Show the {fertility rate} of {Japan} and {@(Bottom 5 countries?)} in {@(Y - 2)} and {@(Y - 1)}.",
            "rephrase": "The {fertility rate} of {Japan} in {@(Y - 1)} is lower than @{(Y - 2)}, and is the lowest among {@(Bottom 5 countries?)} in {@(Y - 1)}."
        }"""}]},

    {"messages": [
        {"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
        {"role": "user", "content": "the total export began to sink more quickly in the 2000s during the financial crises"},
        {"role": "assistant", "content": """{
            "country": ["World"],
            "value": [{
                "raw": "total export",
                "rephrase": "total export"
            }],
            "datetime": ["2000 - 2009", "1990 - 1999"],
            "vis": "Show the {total export} of {World} from {1990} to {1999} and from {2000} to {2009}.",
            "rephrase": "The {total export} of {World} began to sink more quickly from {2000} to {2009} compared to {1990} to {1999}."
        }"""}]},

            
]

import json
# # save pro to query_generation_2.json
# with open("fewshots/claim_tagging_2.json", "w") as f:
#     json.dump(sentence_tagging_2, f, indent=4)

with open("finetune/claim_tagging.jsonl", "w") as outfile:
    for entry in train_data:
        json.dump(entry, outfile)
        outfile.write('\n')