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
        

    clf = DecisionTreeClassifier(criterion="entropy", random_state=1234)
    score= cross_val_score(clf, X.values, y, cv=5)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6) 
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    print(score)

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
   
    rows = [ "home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7", 
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset = rows, inplace = True)

    y= []
    stats= pd.DataFrame()
    (n,d)= match_data.shape

    for match in range(n):
        label = get_match_label(match_data.iloc[match])
        if(label!=0):
            y.append(label)
            stats = stats.append(get_fifa_stats(match_data.iloc[match], player_attributes))

    X = stats.drop(columns=['match_api_id'])
    y = pd.DataFrame(y)

    X.to_pickle("data/X_players_score.pkl")
    y.to_pickle("data/y_players_score.pkl")
    return X, y

def get_match_label(match):
    ''' Derives a label for a given match. '''
    
    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']
     
    # label = 0
    # label = pd.DataFrame()
    # label.loc[0,'match_api_id'] = match['match_api_id'] 

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
    # return label.loc[0]

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