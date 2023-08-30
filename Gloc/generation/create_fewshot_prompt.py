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

import json
# save pro to query_generation_2.json
with open("fewshots/claim_extraction.json", "w") as f:
    json.dump(claims_extraction_with_context, f, indent=4)