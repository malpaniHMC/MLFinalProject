import sqlite3
import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

conn = sqlite3.connect('database.sqlite')

def getData():
    print("Getting Data.....................")
    with sqlite3.connect('database.sqlite') as conn:
        # countries = pd.read_sql_query("SELECT * from Country", conn)
        print("Geting match data.......")
        match_data = pd.read_sql("SELECT * from Match", conn)
        # leagues = pd.read_sql_query("SELECT * from League", conn)
        # teams = pd.read_sql_query("SELECT * from Team", conn)
        # player = pd.read_sql_query("SELECT * from Player",conn)
        print("Geting player data.......")
        player_attributes = pd.read_sql("SELECT * from Player_Attributes",conn)
        # sequence = pd.read_sql("SELECT * from sqlite_sequence",conn)
        print("Geting team data.......")
        team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",conn)
   
    rows = [ "home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7", 
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset = rows, inplace = True)

    y= []
    X = pd.DataFrame()
    (n,d)= match_data.shape

    for match in range(n):
        if (match % 1000) == 0:
            print('Match: {}'.format(match))
        label = get_match_label(match_data.iloc[match])
        if(label!=0):
            y.append(label)
            playerStats = get_players_stats(match_data.iloc[match], player_attributes)
            teamStats = get_team_stats(match_data.iloc[match], team_attributes)
            stats = pd.concat([playerStats, teamStats], axis = 1)
            X = X.append(stats, ignore_index = True)

    X = X.drop(columns=['match_api_id'])
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

def get_players_stats(match, player_stats):
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
            finishing = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])
            finishing = pd.Series(current_stats.loc[0,"finishing"])

        features = [("overall_rating", overall_rating), ("finishing",finishing)]
        for name,feature in features:
             #Rename stat
            name = "{}_{}".format(player, name)
            names.append(name)

            #Aggregate stats
            player_stats_new = pd.concat([player_stats_new, feature], axis = 1)
       
    player_stats_new.columns = names        
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats    
    return player_stats_new    

def get_team_stats(match, team_attributes):
    match_id =  match.match_api_id
    date = match['date']
    teams = ["home_team", "away_team"]
    team_stats_new = pd.DataFrame()
    names = []
    for team in teams:
        #Get team ID
        team_id = match["{}_api_id".format(team)]
        # print(team_id)
        #Get team stats 
        stats = team_attributes[team_attributes.team_api_id == team_id]
        # print(stats)
        #Identify current stats       
        current_stats = stats.sort_values(by = 'date', ascending = False)[:1]
        # print(current_stats)
        if np.isnan(team_id):
            chanceCreatingScore = pd.Series(0)
            defensePressure = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            chanceCreatingScore = pd.Series(current_stats.loc[0, "chanceCreationShooting"])
            defensePressure = pd.Series(current_stats.loc[0, "defencePressure"])

        features = [("chanceCreationShooting", chanceCreatingScore), ("defencePressure",defensePressure)]
        for name,feature in features:
             #Rename stat
            name = "{}_{}".format(team, name)
            names.append(name)

            #Aggregate stats
            team_stats_new = pd.concat([team_stats_new, feature], axis = 1)
    team_stats_new.columns = names        
    # team_stats_new['match_api_id'] = match_id

    team_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats    
    return team_stats_new

def main(args): 
    if args.generate:
        print("Generating feature matrix")
        X,y = getData()
    else:
        try:
            X = pd.read_pickle("data/X_players_score.pkl")
            y = pd.read_pickle("data/y_players_score.pkl")
            print("Using existing feature matrix")
        except:
            print("Generating feature matrix")
            X,y = getData()
    print("Classifying with Deicison tree.....................")
    clf = DecisionTreeClassifier(criterion="entropy", random_state=1234)
    score = cross_val_score(clf, X.values, y, cv=5)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6) 
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-generate", action="store_true",
                        help = "Regenerate Data")
    args = parser.parse_args()
    main(args)