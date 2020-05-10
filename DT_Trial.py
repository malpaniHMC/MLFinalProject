import sqlite3
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

conn = sqlite3.connect('database.sqlite')

def main(): 
    try:
        X = pd.read_pickle("data/X_players_score.pkl")
        y = pd.read_pickle("data/y_players_score.pkl")
    except:
        print("error")
        X,y = getData()
    
    print("X")
    print(X)
    print("y")
    print(np.unique(y.values))

    clf = DecisionTreeClassifier(criterion="entropy", random_state=1234)
    score= cross_val_score(clf, X.values, y.values, cv=5)
    print("DT", score)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6) 
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SVC", score)

def getData():
    with sqlite3.connect('database.sqlite') as con:
        countries = pd.read_sql_query("SELECT * from Country", con)
        match_data = pd.read_sql("SELECT * from Match", con)
        leagues = pd.read_sql_query("SELECT * from League", con)
        teams = pd.read_sql_query("SELECT * from Team", con)
        player = pd.read_sql_query("SELECT * from Player",con)
        player_attributes = pd.read_sql("SELECT * from Player_Attributes",con)
        sequence = pd.read_sql("SELECT * from sqlite_sequence",con)
        team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)
   
    rows = ["home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7", 
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11", "home_team_api_id", "away_team_api_id"]
    match_data.dropna(subset = rows, inplace = True)

    y= []
    stats= pd.DataFrame()
    (n,d)= match_data.shape

    for match_indx in range(n):
        match = match_data.iloc[match_indx]
        label = get_match_label(match)
        team_stats = get_team_stats(match, team_attributes)
        player_stats = get_fifa_stats(match_data.iloc[match_indx], player_attributes)
        # and team_attr.notna().all(axis=None)z
        if(label!=0 and team_stats is not None):
            y.append(label)
            # print("Team Stats:")
            # print(team_stats.shape)
            # print(team_stats.head())
            # print("Player Stats:")
            # print(player_stats.shape)
            # print(player_stats.head())
            toAppend = pd.merge(player_stats, team_stats, how="left")
            # print("To Append:")
            # print(toAppend.head())
            stats= stats.append(toAppend)
            # print("STATS:")
            # print(stats.shape)
            # print(stats.head())
        
    
    X = stats.drop(columns=['match_api_id'])
    y = pd.DataFrame(y)

    X.to_pickle("data/X_players_score.pkl")
    y.to_pickle("data/y_players_score.pkl")
    return X, y

def get_team_stats(match, team_attributes):
    if(match.home_team_api_id is np.nan or match.away_team_api_id is np.nan): 
        return None 
    date = match["date"]
    team_stats = pd.DataFrame()
    team_attrs = ['date', 'buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 
                'defencePressure', 'defenceAggression', 'defenceTeamWidth']   
    # team_attrs = ['date', 'buildUpPlaySpeed']   
    
    home_team_stats = team_attributes[team_attributes.team_api_id == match['home_team_api_id']]
    home_team_stats = home_team_stats[team_attrs]
    current_home_team_stats = home_team_stats[home_team_stats.date < date].sort_values(by = 'date', ascending = False)[:1]
    team_stats = current_home_team_stats[team_attrs].drop(columns=['date']).add_prefix("home_")
    
    away_team_stats = team_attributes[team_attributes.team_api_id == match['away_team_api_id']]
    away_team_stats = away_team_stats[team_attrs]
    current_away_team_stats = away_team_stats[away_team_stats.date < date].sort_values(by = 'date', ascending = False)[:1]
    team_stats = pd.concat([team_stats,current_home_team_stats[team_attrs].drop(columns=['date']).add_prefix("away_").reindex()], axis=1)
    team_stats["match_api_id"]= match.match_api_id
    if(not team_stats.empty):
        # print(team_stats)
        return team_stats 
    return None


# Code based on the work done by Pavan Raj on Kaggle
def get_match_label(match):
    ''' Derives a label for a given match. '''
    
    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    #Identify match label  
    if home_goals > away_goals:
        return 1
        # label.loc[0,'label'] = "Win"
    if home_goals == away_goals:
        return 0
        # label.loc[0,'label'] = "Draw"
    if home_goals < away_goals:
        return -1
        # label.loc[0,'label'] = "Defeat"

    return 0        

# Code based on the work done by Pavan Raj on Kaggle
def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''    
    
    #Define variables
    match_id =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []
    
    #Loop through all players
    for player in players:   
            
        #Get player ID
        player_id = match[player]
        
        #Get player stats 
        stats = player_stats[player_stats.player_api_id == player_id]
            
        #Identify current stats       
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]
        
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        #Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)
            
        #Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)
    
    player_stats_new.columns = names        
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats    
    return player_stats_new    

main()