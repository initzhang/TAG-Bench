"""
extract (1) relational query instruction type (2) query template (3) associated dataframe columns 
the instruction is combined with corresponding template such as https://github.com/guestrin-lab/lotus/blob/97618552e3d3e8b6c85ef16524cc7f10a526bdb0/lotus/templates/task_instructions.py#L125 and finally becomes the prompt
"""
import argparse
import json
import os
import re
import time

import pandas as pd

############################################################################################################
###################################### Match based queries #################################################
############################################################################################################


def pipeline_0():
    query = "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?"
    answer = 71
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    scores_df = scores_df[scores_df["AvgScrMath"] > 560]
    return "filter", "{cname} is a county in the Bay Area", scores_df[["cname"]]
    scores_df = scores_df.sem_filter("{cname} is a county in the Bay Area")
    prediction = len(scores_df)
    return prediction, answer


def pipeline_1():
    query = (
        "What is the telephone number for the school with the lowest average score in reading in Southern California?"
    )
    answer = "(562) 944-0033"
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    return "filter", "{cname} is a county name in Southern California", scores_df[["cname"]]
    scores_df = scores_df.sem_filter("{cname} is a county name in Southern California")
    scores_df = scores_df.loc[[scores_df["AvgScrRead"].idxmin()]]

    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = merged_df.Phone.values[0]
    return prediction, answer


def pipeline_3():
    query = "How many test takers are there at the school/s in a county with population over 2 million?"
    answer = 244742
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    unique_counties = pd.DataFrame(schools_df["County"].unique(), columns=["County"])
    return "map", "What is the population of {County} in California? Answer with only the number without commas. Respond with your best guess.", unique_counties[["County"]]
    unique_counties = unique_counties.sem_map(
        "What is the population of {County} in California? Answer with only the number without commas. Respond with your best guess."
    )
    counties_over_2m = set()
    for _, row in unique_counties.iterrows():
        try:
            if int(re.findall(r"\d+", row._map)[-1]) > 2000000:
                counties_over_2m.add(row.County)
        except:
            pass

    schools_df = schools_df[schools_df["County"].isin(counties_over_2m)]
    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = int(merged_df["NumTstTakr"].sum())
    return prediction, answer


def pipeline_4():
    query = "What is the grade span offered in the school with the highest longitude in cities in that are part of the 'Silicon Valley' region?"
    answer = "K-5"
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    unique_cities = pd.DataFrame(schools_df["City"].unique(), columns=["City"])
    return "filter", "{City} is a city in the Silicon Valley region", unique_cities[["City"]]
    unique_cities = unique_cities.sem_filter("{City} is a city in the Silicon Valley region")
    schools_df = schools_df[schools_df["City"].isin(unique_cities["City"])]
    schools_df = schools_df.sort_values(by=["Longitude"], key=abs, ascending=False).head(1)
    prediction = schools_df["GSoffered"].tolist()[0]
    return prediction, answer


def pipeline_5():
    query = "What are the two most common first names among the female school administrators?"
    answer = ["Jennifer", "Lisa"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")

    schools_df = (
        schools_df.groupby("AdmFName1").size().reset_index(name="count").sort_values("count", ascending=False).head(20)
    )
    return "filter", "{AdmFName1} is a female first name", schools_df[["AdmFName1"]]
    schools_df = schools_df.sem_filter("{AdmFName1} is a female first name")
    prediction = schools_df["AdmFName1"].tolist()[:2]
    return prediction, answer


def pipeline_6():
    query = "Among the posts owned by csgillespie, how many of them are root posts and mention academic papers?"
    answer = 4
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    posts_df = posts_df[posts_df["ParentId"].isna()]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId")
    return "filter", "{Body} mentions academic papers", merged_df[["Body"]]
    merged_df = merged_df.sem_filter("{Body} mentions academic papers")

    prediction = len(merged_df)
    return prediction, answer


def pipeline_8():
    query = "How many of the comments with a score of 17 are about statistics?"
    answer = 4
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    comments_df = comments_df[comments_df["Score"] == 17]
    return "filter", "{Text} is about statistics", comments_df[["Text"]]
    comments_df = comments_df.sem_filter("{Text} is about statistics")
    prediction = len(comments_df)

    return prediction, answer


def pipeline_10():
    query = "Of the posts with views above 80000, how many discuss the R programming language?"
    answer = 3
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    posts_df = posts_df[posts_df["ViewCount"] > 80000]
    return "filter", "{Body} discusses the R programming language", posts_df[["Body"]]
    posts_df = posts_df.sem_filter("{Body} discusses the R programming language")
    prediction = len(posts_df)

    return prediction, answer


def pipeline_11():
    query = "Please give the names of the races held on the circuits in the middle east."
    answer = [
        "Bahrain Grand Prix",
        "Turkish Grand Prix",
        "Abu Dhabi Grand Prix",
        "Azerbaijan Grand Prix",
        "European Grand Prix",
    ]
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    return "filter", "{country} is in the Middle East", circuits_df[["country"]]
    return circuits_df.shape[0]
    circuits_df = circuits_df.sem_filter("{country} is in the Middle East")

    merged_df = pd.merge(circuits_df, races_df, on="circuitId", suffixes=["_circuit", "_race"]).drop_duplicates(
        subset="name_race"
    )
    prediction = merged_df["name_race"].tolist()

    return prediction, answer


def pipeline_13():
    query = "How many Asian drivers competed in the 2008 Australian Grand Prix?"
    answer = 2

    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    return "filter", "{nationality} is Asian", drivers_df[["nationality"]]
    drivers_df = drivers_df.sem_filter("{nationality} is Asian")
    races_df = races_df[(races_df["name"] == "Australian Grand Prix") & (races_df["year"] == 2008)]
    merged_df = pd.merge(pd.merge(races_df, results_df, on="raceId"), drivers_df, on="driverId")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_18():
    query = "List the football player with a birthyear of 1970 who is an Aquarius"
    answer = "Hans Vonk"
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    players_df = players_df[players_df["birthday"].str.startswith("1970")]
    return "filter", "Someone born on {birthday} would be an Aquarius", players_df[["birthday"]]
    players_df = players_df.sem_filter("Someone born on {birthday} would be an Aquarius")
    prediction = players_df["player_name"].values[0]
    return prediction, answer


def pipeline_19():
    query = "Please list the league from the country which is landlocked."
    answer = "Switzerland Super League"
    leagues_df = pd.read_csv("../pandas_dfs/european_football_2/League.csv")
    countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
    return "filter", "{name} is landlocked", countries_df[["name"]]
    countries_df = countries_df.sem_filter("{name} is landlocked")
    merged_df = pd.merge(
        leagues_df, countries_df, left_on="country_id", right_on="id", suffixes=["_league", "_country"]
    )
    prediction = merged_df["name_league"].values[0]

    return prediction, answer


def pipeline_20():
    query = "How many matches in the 2008/2009 season were held in countries where French is an official language?"
    answer = 866
    matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")
    countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
    matches_df = matches_df[matches_df["season"] == "2008/2009"]
    return "filter", "{name} has French as an official language", countries_df[["name"]]
    countries_df = countries_df.sem_filter("{name} has French as an official language")
    merged_df = pd.merge(matches_df, countries_df, left_on="country_id", right_on="id")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_21():
    query = "Of the top three away teams that scored the most goals, which one has the most fans?"
    answer = "FC Barcelona"
    teams_df = pd.read_csv("../pandas_dfs/european_football_2/Team.csv")
    matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")

    merged_df = pd.merge(matches_df, teams_df, left_on="away_team_api_id", right_on="team_api_id")
    merged_df = (
        merged_df.sort_values("away_team_goal", ascending=False).drop_duplicates(subset="team_long_name").head(3)
    )
    """
    naive pairwise comparison, N samples lead to ≈(N^2)/2 LM calls
    """
    return "topk", "What {team_long_name} has the most fans?", merged_df[["team_long_name"]]
    #return prediction.shape[0] ** 2
    prediction = merged_df.sem_topk("What {team_long_name} has the most fans?", 1).team_long_name.values[0]
    return prediction, answer


def pipeline_24():
    query = "Which year recorded the most gas use paid in the higher value currency?"
    answer = 2013
    customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
    yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")

    unique_currencies = customers_df["Currency"].unique()
    return "topk", "What {Currency} is the highest value currency?", pd.DataFrame(unique_currencies, columns=["Currency"])
    return pd.DataFrame(unique_currencies, columns=["Currency"]).shape[0] ** 2
    most_value = (
        pd.DataFrame(unique_currencies, columns=["Currency"])
        .sem_topk("What {Currency} is the highest value currency?", 1)
        .Currency.values[0]
    )
    customers_df = customers_df[customers_df["Currency"] == most_value]

    yearmonth_df["year"] = yearmonth_df["Date"] // 100
    merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
    merged_df = merged_df.groupby("year")["Consumption"].sum().reset_index()
    merged_df = merged_df.sort_values("Consumption", ascending=False)
    prediction = int(merged_df["year"].values[0])

    return prediction, answer


def pipeline_108():
    query = "Among the posts that were voted by user 1465, determine if the post is relevant to machine learning. Respond with YES if it is and NO if it is not."
    answer = ["YES", "YES", "YES"]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    votes_df = pd.read_csv("../pandas_dfs/codebase_community/votes.csv")
    votes_df = votes_df[votes_df["UserId"] == 1465]
    merged_df = pd.merge(posts_df, votes_df, left_on="Id", right_on="PostId")
    return "map", "{Body} is relevant to machine learning. Answer with YES if it is and NO if it is not.", merged_df[["Body"]]
    merged_df = merged_df.sem_map(
        "{Body} is relevant to machine learning. Answer with YES if it is and NO if it is not."
    )
    prediction = merged_df._map.tolist()
    return prediction, answer


def pipeline_109():
    query = "Extract the statistical term from the post titles which were edited by Vebjorn Ljosa."
    answer = ["beta-binomial distribution", "AdaBoost", "SVM", "Kolmogorov-Smirnov statistic"]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    merged_df = pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
    merged_df = merged_df[merged_df["DisplayName"] == "Vebjorn Ljosa"]
    return "map", "Extract the statistical term from {Title}. Respond with only the statistical term.", merged_df[["Title"]]
    merged_df = merged_df.sem_map("Extract the statistical term from {Title}. Respond with only the statistical term.")
    prediction = merged_df._map.tolist()
    return prediction, answer


def pipeline_110():
    query = "List the Comment Ids of the positive comments made by the top 5 newest users on the post with the title 'Analysing wind data with R'"
    answer = [11449]
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = posts_df[posts_df["Title"] == "Analysing wind data with R"]
    merged_df = pd.merge(comments_df, posts_df, left_on="PostId", right_on="Id", suffixes=["_comment", "_post"])
    merged_df = pd.merge(merged_df, users_df, left_on="UserId", right_on="Id", suffixes=["_merged", "_user"])
    merged_df = merged_df.sort_values(by=["CreationDate_user"], ascending=False).head(5)
    return "filter", "The sentiment of {Text} is positive", merged_df[["Text"]]
    merged_df = merged_df.sem_filter("The sentiment of {Text} is positive")
    prediction = merged_df.Id_comment.tolist()
    return prediction, answer


def pipeline_111():
    query = 'For the post from which the tag "bayesian" is excerpted from, identify whether the body of the post is True or False. Answer with True or False ONLY.'
    answer = "TRUE"
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
    tags_df = tags_df[tags_df["TagName"] == "bayesian"]
    merged_df = pd.merge(tags_df, posts_df, left_on="ExcerptPostId", right_on="Id")
    return "map", "Determine whether the content in {Body} is true. Respond with only TRUE or FALSE.", merged_df[["Body"]]
    prediction = merged_df.sem_map("Determine whether the content in {Body} is true. Respond with only TRUE or FALSE.")[
        "_map"
    ].values[0]
    return prediction, answer


############################################################################################################
###################################### Comparison based queries ############################################
############################################################################################################

def pipeline_29():
    query = "What is the difference in gas consumption between customers who pay using the currency of the Czech Republic and who pay the currency of European Union in 2012, to the nearest integer?"
    answer = 402524570
    customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
    yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")

    countries = {"Area": ["Czech Republic", "European Union"]}
    countries_df = pd.DataFrame(countries)
    return "map", "Given {Area}, return the 3 letter currency code for the area. Answer with the code ONLY.", countries_df
    currency_df = countries_df.sem_map(
        "Given {Area}, return the 3 letter currency code for the area. Answer with the code ONLY.", suffix="currency"
    )
    currencies = currency_df["currency"].values.tolist()

    yearmonth_df = yearmonth_df[yearmonth_df["Date"] // 100 == 2012]

    merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
    first_df = merged_df[merged_df["Currency"] == currencies[0]]
    second_df = merged_df[merged_df["Currency"] == currencies[1]]

    prediction = round(first_df["Consumption"].sum() - second_df["Consumption"].sum())

    return prediction, answer


def pipeline_36():
    query = "Among all European Grand Prix races, what is the percentage of the races were hosted in the country where the Bundesliga happens, to the nearest whole number?"
    answer = 52
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    races_df = races_df[races_df["name"] == "European Grand Prix"]
    merged_df = pd.merge(circuits_df, races_df, on="circuitId")
    denom = len(merged_df)

    return "filter", "{country} is where the Bundesliga happens", merged_df[["country"]]
    return denom
    merged_df = merged_df.sem_filter("{country} is where the Bundesliga happens")
    numer = len(merged_df)
    prediction = int(numer * 100 / denom)
    return prediction, answer


def pipeline_37():
    query = "From 2010 to 2015, what was the average overall rating, rounded to the nearest integer, of players who are higher than 170 and shorter than Michael Jordan?"
    answer = 69
    jordan_df = pd.DataFrame({"Name": ["Michael Jordan"]})
    return "map", "Given {Name}, provide the height in cm. Answer with ONLY the number to one decimal place.", jordan_df
    jordan_df = jordan_df.sem_map(
        "Given {Name}, provide the height in cm. Answer with ONLY the number to one decimal place.", suffix="height"
    )
    jordan_height = float(jordan_df["height"].values.tolist()[0])

    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    players_df = players_df[players_df["height"] > 170]
    players_df = players_df[players_df["height"] < jordan_height]

    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    attributes_df["year"] = attributes_df["date"].str[:4].astype(int)
    attributes_df = attributes_df[attributes_df["year"] >= 2010]
    attributes_df = attributes_df[attributes_df["year"] <= 2015]

    merged_df = pd.merge(players_df, attributes_df, on="player_api_id")
    prediction = round(merged_df["overall_rating"].mean())

    return prediction, answer


def pipeline_38():
    query = "Among the drivers that finished the race in the 2008 Australian Grand Prix, how many debuted earlier than Lewis Hamilton?"
    answer = 3
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")

    races_df = races_df[races_df["name"] == "Australian Grand Prix"]
    races_df = races_df[races_df["year"] == 2008]
    results_df = results_df[results_df["time"].notnull()]
    return "map", "What year did driver {forename} {surname} debut in Formula 1? Answer with the year ONLY.", drivers_df[["forename","surname"]]
    drivers_df = drivers_df.sem_map(
        "What year did driver {forename} {surname} debut in Formula 1? Answer with the year ONLY.", suffix="debut"
    )
    drivers_df["debut"] = pd.to_numeric(drivers_df["debut"], errors="coerce")
    drivers_df = drivers_df.dropna(subset=["debut"])

    merged_df = pd.merge(results_df, races_df, on="raceId").merge(drivers_df, on="driverId")
    merged_df = merged_df[merged_df["year"] > merged_df["debut"]]

    prediction = len(merged_df)

    return prediction, answer


def pipeline_48():
    query = "Which of these circuits is located closer to a capital city, Silverstone Circuit, Hockenheimring or Hungaroring?"
    answer = "Hungaroring"
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["name"].isin(["Silverstone Circuit", "Hockenheimring", "Hungaroring"])]

    return "topk", "What circuit, named {name} is located closer to a capital city?", circuits_df[["name"]]
    return circuits_df.shape[0]**2
    prediction = circuits_df.sem_topk("What circuit, named {name} is located closer to a capital city?", 1).name.values[
        0
    ]

    return prediction, answer


def pipeline_49():
    query = "Which race was Alex Yoong in when he was in the top half of finishers?"
    answer = "Australian Grand Prix"
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    drivers_df = drivers_df[(drivers_df["forename"] == "Alex") & (drivers_df["surname"] == "Yoong")]
    driverStandings_df = pd.read_csv("../pandas_dfs/formula_1/driverStandings.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(drivers_df, driverStandings_df, on="driverId").merge(races_df, on="raceId")

    return "filter", "The {position} is in the top half of number racers in a formula 1 race.", merged_df[["position"]]
    return merged_df.shape[0]
    prediction = merged_df.sem_filter(
        "The {position} is in the top half of number racers in a formula 1 race."
    ).name.values[0]

    return prediction, answer


############################################################################################################
###################################### Ranking based queries ###############################################
############################################################################################################


def pipeline_50():
    query = "Among the magnet schools with SAT test takers of over 500, which school name sounds most futuristic?"
    answer = "Polytechnic High"

    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    schools_df = schools_df[schools_df["Magnet"] == 1]
    satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    satscores_df = satscores_df[satscores_df["NumTstTakr"] > 500]
    merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
    return "topk", "What {School} sounds most futuristic?", merged_df[["School"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {School} sounds most futuristic?", 1).School.values[0]

    return prediction, answer


def pipeline_51():
    query = "Of the 5 posts wih highest popularity, list their titles in order of most technical to least technical."
    answer = [
        "How to interpret and report eta squared / partial eta squared in statistically significant and non-significant analyses?",
        "How to interpret F- and p-value in ANOVA?",
        "What is the meaning of p values and t values in statistical tests?",
        "How to choose between Pearson and Spearman correlation?",
        "How do I get the number of rows of a data.frame in R?",
    ]

    posts_df = (
        pd.read_csv("../pandas_dfs/codebase_community/posts.csv").sort_values(by=["ViewCount"], ascending=False).head(5)
    )

    return "topk", "What {Title} is most technical?", posts_df[["Title"]]
    return posts_df.shape[0] ** 2
    prediction = posts_df.sem_topk("What {Title} is most technical?", 5).Title.values.tolist()

    return prediction, answer


def pipeline_52():
    query = "What are the Post Ids of the top 2 posts in order of most grateful comments received on 9-14-2014"
    answer = [115372, 115254]

    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    comments_df = comments_df[comments_df["CreationDate"].str.startswith("2014-09-14")]
    merged_df = pd.merge(posts_df, comments_df, left_on="Id", right_on="PostId")
    return "filter", "The sentiment on {Text} is that of someone being grateful.", merged_df[["Text"]]
    return merged_df.shape[0]
    merged_df = merged_df.sem_filter("The sentiment on {Text} is that of someone being grateful.")
    merged_df = merged_df.groupby("Id_x").size().sort_values(ascending=False)
    prediction = list(merged_df.index[:2])

    return prediction, answer


def pipeline_53():
    query = "For the post owned by csgillespie with the highest popularity, what is the most sarcastic comment?"
    answer = "That pirates / global warming chart is clearly cooked up by conspiracy theorists - anyone can see they have deliberately plotted even spacing for unequal time periods to avoid showing the recent sharp increase in temperature as pirates are almost entirely wiped out. We all know that as temperatures rise it makes the rum evaporate and pirates cannot survive those conditions."

    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = (
        pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
        .sort_values(by=["ViewCount"], ascending=False)
        .head(1)
    )

    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    merged_df_with_comments = pd.merge(merged_df, comments_df, left_on="Id_x", right_on="PostId")
    return "topk", "What {Text} is most sarcastic?", merged_df_with_comments[["Text"]]
    return merged_df_with_comments.shape[0] ** 2
    prediction = merged_df_with_comments.sem_topk("What {Text} is most sarcastic?", 1).Text.values[0]

    return prediction, answer


def pipeline_54():
    query = "Among the top 10 most popular tags, which is the least related to statistics?"
    answer = "self-study"
    tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
    tags_df = tags_df.sort_values("Count", ascending=False)
    tags_df = tags_df.head(10)
    return "topk", "{TagName} is the least related to statistics?", tags_df[["TagName"]]
    return tags_df.shape[0] ** 2 
    prediction = tags_df.sem_topk("{TagName} is the least related to statistics?", 1).TagName.values[0]
    return prediction, answer


def pipeline_55():
    query = "Of the top 10 most favorited posts, what is the Id of the most lighthearted post?"
    answer = 423
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    posts_df = posts_df.sort_values("FavoriteCount", ascending=False).head(10)
    return "topk", "What {Body} is most lighthearted?", posts_df[["Body"]]
    return posts_df.shape[0] ** 2
    prediction = posts_df.sem_topk("What {Body} is most lighthearted?", 1).Id.values[0]
    prediction = int(prediction)

    return prediction, answer


def pipeline_56():
    query = "Among the posts owned by a user over 65 with a score of over 10, what are the post id's of the top 2 posts made with the least expertise?"
    answer = [8485, 15670]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df[users_df["Age"] > 65]
    posts_df = posts_df[posts_df["Score"] > 10]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
    return "topk", "What {Body} is made with the least expertise?", merged_df[["Body"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {Body} is made with the least expertise?", 2).Id_posts.values.tolist()

    return prediction, answer


def pipeline_57():
    query = "Among the badges obtained by csgillespie in 2011, which is the most creatively named?"
    answer = "Strunk & White"
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId").drop_duplicates("Name")
    return "topk", "What {Name} is most creative?", merged_df[["Name"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {Name} is most creative?", 1).Name.values[0]

    return prediction, answer


def pipeline_58():
    query = "Of the posts owned by Yevgeny, what are the id's of the top 3 most pessimistic?"
    answer = [23819, 24216, 35748]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "Yevgeny"]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
    return "topk", "What {Body} is most pessimistic?", merged_df[["Body"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {Body} is most pessimistic?", 3).Id_posts.values.tolist()

    return prediction, answer


def pipeline_59():
    query = "Of the top 10 players taller than 180 ordered by average heading accuracy, what are the top 3 most unique sounding names?"
    answer = ["Naldo", "Per Mertesacker", "Didier Drogba"]
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    players_df = players_df[players_df["height"] > 180]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id", suffixes=["_players", "_attributes"])
    merged_df = merged_df.groupby("player_api_id")
    merged_df = merged_df.agg({"heading_accuracy": "mean", "player_name": "first"}).reset_index()
    merged_df = merged_df.sort_values("heading_accuracy", ascending=False).head(10)
    return "topk", "What {player_name} is most unique sounding?", merged_df[["player_name"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {player_name} is most unique sounding?", 3).player_name.values.tolist()

    return prediction, answer


def pipeline_60():
    query = "Out of users that have obtained at least 200 badges, what are the top 2 display names that seem most based off a real name?"
    answer = ["Jeromy Anglim", "Glen_b"]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
    merged_df = merged_df.groupby("DisplayName").filter(lambda x: len(x) >= 200).drop_duplicates(subset="DisplayName")
    return "topk", "What {DisplayName} seems most based off a real name?", merged_df[["DisplayName"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk(
        "What {DisplayName} seems most based off a real name?", 2
    ).DisplayName.values.tolist()
    return prediction, answer


def pipeline_61():
    query = "Of the cities containing exclusively virtual schools which are the top 3 safest places to live?"
    answer = ["Thousand Oaks", "Simi Valley", "Westlake Village"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    schools_df = schools_df[schools_df["Virtual"] == "F"].drop_duplicates(subset="City")
    return "topk", "What {City} is the safest place to live?", schools_df[["City"]]
    return schools_df.shape[0] ** 2
    prediction = schools_df.sem_topk("What {City} is the safest place to live?", 3).City.values.tolist()
    return prediction, answer


def pipeline_62():
    query = "List the cities containing the top 5 most enrolled schools in order from most diverse to least diverse. "
    answer = ["Long Beach", "Paramount", "Granada Hills", "Temecula", "Carmichael"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=False).head(5)
    return "topk", "What {City} is the most diverse?", merged_df[["City"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {City} is the most diverse?", 5).City.values.tolist()
    return prediction, answer


def pipeline_63():
    query = "Please list the top three continuation schools with the lowest eligible free rates for students aged 5-17 and rank them based on the overall affordability of their respective cities."
    answer = ["Del Amigo High (Continuation)", "Rancho del Mar High (Continuation)", "Millennium High Alternative"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    frpm_df = frpm_df[frpm_df["Educational Option Type"] == "Continuation School"]
    frpm_df["frpm_rate"] = frpm_df["Free Meal Count (Ages 5-17)"] / frpm_df["Enrollment (Ages 5-17)"]
    frpm_df = frpm_df.sort_values("frpm_rate", ascending=True).head(3)
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    return "topk", "What {City} is the most affordable?", merged_df[["City"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {City} is the most affordable?", 3).School.values.tolist()
    return prediction, answer


def pipeline_64():
    query = "Of the schools with the top 3 SAT excellence rate, order their counties by academic reputation from strongest to weakest."
    answer = "Santa Clara County"
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    satscores_df["excellence_rate"] = satscores_df["NumGE1500"] / satscores_df["NumTstTakr"]
    satscores_df = satscores_df.sort_values("excellence_rate", ascending=False).head(3)
    merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
    return "topk", "What {County} has the strongest academic reputation?", merged_df[["County"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {County} has the strongest academic reputation?", 3).County.values[0]
    return prediction, answer


def pipeline_65():
    query = "Among the cities with the top 10 lowest enrollment for students in grades 1 through 12, which are the top 2 most popular cities to visit?"
    answer = ["Death Valley", "Shaver Lake"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    merged_df = merged_df.groupby("City").agg({"Enrollment (K-12)": "sum"}).reset_index()
    merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=True).head(10)
    return "topk", "What {City}/location in California is the most popular to visit?", merged_df[["City"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk(
        "What {City}/location in California is the most popular to visit?", 2
    ).City.values.tolist()
    return prediction, answer


def pipeline_106():
    query = "Of the top 5 users with the most views, who has their social media linked in their AboutMe section?"
    answer = "whuber"
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df.sort_values("Views", ascending=False).head(5)
    return users_df.shape[0]
    prediction = users_df.sem_filter("The {AboutMe} contains a link to social media.").DisplayName.values[0]
    return prediction, answer


def pipeline_107():
    query = "Of all the comments commented by the user with a username of Harvey Motulsky and with a score of 5, rank the post ids in order of most helpful to least helpful."
    answer = [89457, 64710, 4945]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    users_df = users_df[users_df["DisplayName"] == "Harvey Motulsky"]
    comments_df = comments_df[comments_df["Score"] == 5]
    merged_df = pd.merge(users_df, comments_df, left_on="Id", right_on="UserId")
    return "topk", "What {Text} is most helpful?", merged_df[["Text"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {Text} is most helpful?", 3).PostId.values.tolist()
    return prediction, answer


def pipeline_952():
    query = "Of the constructors that have been ranked 1 in 2014, which has the most prestige"
    answer = "Ferrari"
    constructors_df = pd.read_csv("../pandas_dfs/formula_1/constructors.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(results_df, constructors_df, on="constructorId", suffixes=["_results", "_constructors"])
    merged_df = pd.merge(merged_df, races_df, on="raceId", suffixes=["_merged", "_races"])
    merged_df = merged_df[(merged_df["rank"] == 1) & (merged_df["year"] == 2014)].drop_duplicates(
        subset="constructorId"
    )
    merged_df = merged_df.rename(columns={"name_merged": "name"})
    return "topk", "What {name} is most prestigious?", merged_df[["name"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {name} is most prestigious?", 1).name.values[0]
    return prediction, answer


def pipeline_1000():
    query = "Of the 5 racetracks that hosted the most recent races, rank the locations by distance to the equator."
    answer = ["Mexico City", "Sao Paulo", "Abu Dhabi", "Austin", "Suzuka"]
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(circuits_df, races_df, on="circuitId")
    merged_df = merged_df.sort_values("date", ascending=False).head(5)
    return  "topk", "What {location} is closest to the equator?", merged_df[["location"]]
    return merged_df.shape[0] ** 2
    prediction = merged_df.sem_topk("What {location} is closest to the equator?", 5).location.values.tolist()
    return prediction, answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", required=True, help="List of pipelines to run")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    bird_ids = []
    rel_types = []
    rel_templates = []
    rel_counts = []
    rel_dfs = []
    for i in range(1001):
        try:
            func = eval(f"pipeline_{i}")
            rel_type, rel_template, rel_df = func()
            nrows = rel_df.shape[0]
            if rel_type == "topk":
                counts = int(nrows*(nrows-1)/2)
            else:
                counts = nrows
            print(i, rel_type, counts)
            bird_ids.append(i)
            rel_types.append(rel_type)
            rel_templates.append(rel_template)
            rel_counts.append(counts)
            rel_dfs.append(rel_df)
        except:
            continue

    meta_info = pd.DataFrame({
        "bird_id": bird_ids, 
        "rel_type": rel_types, 
        "rel_template":rel_templates,
        "rel_counts":rel_counts
        })
    
    meta_info.to_csv("../relational_queries/meta_info.csv",index=False)
    for bird_id, df in zip(bird_ids, rel_dfs):
        df.to_csv(f"../relational_queries/dfs/{bird_id}.csv", index=False)
