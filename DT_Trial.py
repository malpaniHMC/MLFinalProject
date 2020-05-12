import sqlite3
import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

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
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11", "home_team_api_id", "away_team_api_id"]
    match_data.dropna(subset = rows, inplace = True)

    y= []
    stats = pd.DataFrame()
    (n,d)= match_data.shape
    print("Constructing matrix..........")
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
            #print(team_stats.head())
            # print("Player Stats:")
            # print(player_stats.shape)
            # print(player_stats.head())
            toAppend = pd.concat([player_stats, team_stats], axis=1)
            # print("To Append:")
            #print(toAppend.head())
            stats= stats.append(toAppend, ignore_index = True)
            # print("STATS:")
            # print(stats.shape)
            # print(stats.head())
        
    
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
    player_stats_new["home_defense_team"] = (player_stats_new['home_player_2_overall_rating'] + player_stats_new['home_player_3_overall_rating'] + player_stats_new['home_player_4_overall_rating'])/300
    player_stats_new["home_mid_team"] = (player_stats_new['home_player_5_overall_rating'] + player_stats_new['home_player_6_overall_rating'] + player_stats_new['home_player_7_overall_rating'] + player_stats_new['home_player_8_overall_rating'])/400
    player_stats_new["home_attack_team"] = (player_stats_new['home_player_9_overall_rating'] + player_stats_new['home_player_10_overall_rating'] + player_stats_new['home_player_11_overall_rating'])/300

    player_stats_new["away_defense_team"] = (player_stats_new['away_player_2_overall_rating'] + player_stats_new['away_player_3_overall_rating'] + player_stats_new['away_player_4_overall_rating'])/300
    player_stats_new["away_mid_team"] = (player_stats_new['away_player_5_overall_rating'] + player_stats_new['away_player_6_overall_rating'] + player_stats_new['away_player_7_overall_rating'] + player_stats_new['away_player_8_overall_rating'])/400
    player_stats_new["away_attack_team"] = (player_stats_new['away_player_9_overall_rating'] + player_stats_new['away_player_10_overall_rating'] + player_stats_new['away_player_11_overall_rating'])/300


    player_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats    
    return player_stats_new

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
    home_team_stats = current_home_team_stats[team_attrs].drop(columns=['date']).add_prefix("home_").reset_index(drop = True)
    # print('HOME.................................')
    # print(home_team_stats)
    away_team_stats = team_attributes[team_attributes.team_api_id == match['away_team_api_id']]
    away_team_stats = away_team_stats[team_attrs]
    current_away_team_stats = away_team_stats[away_team_stats.date < date].sort_values(by = 'date', ascending = False)[:1]
    away_team_stats = current_away_team_stats[team_attrs].drop(columns=['date']).add_prefix("away_").reset_index(drop = True)
    # print('AWAY.................................')
    # print(away_team_stats)
    team_stats = pd.concat([home_team_stats,away_team_stats], axis=1)
    team_stats["match_api_id"]= match.match_api_id
    if(not home_team_stats.empty and not away_team_stats.empty):
        # print('TEAM................TEAM')
        # print(team_stats)
        return team_stats 
    return None

#adapted from ps4
def get_classifier(clf_str):
    if clf_str == 'dummy':
        clf = DummyClassifier(strategy='stratified')
        param_grid = {}
    elif clf_str == 'rf':
        clf = RandomForestClassifier(max_depth=11, criterion='entropy', random_state=0, n_estimators= 100)
        param_grid = {}
    elif clf_str == 'svc':
        clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6) 
        param_grid = {'C': np.logspace(-5,5,num=11)}
    elif clf_str == 'lr':
        clf = LogisticRegression(penalty='l2', max_iter=10000, solver='lbfgs')
        param_grid = {'C': np.logspace(-5,5,num=11)}
    elif clf_str == 'ada':
        clf = AdaBoostClassifier(n_estimators=100)
        param_grid = {}

    return clf, param_grid

#adapted from ps 4
def get_performance(clf, X,y, numtrials, param_grid):
    train_scores = np.zeros(numtrials)
    test_scores = np.zeros(numtrials)

    for i in range(numtrials):
        print(i)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        
        newClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, iid = False)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', newClf)])
        nested_score = cross_validate(pipe, X, y, cv=outer_cv, return_train_score=True)
        train_scores[i] = np.mean(nested_score['train_score'])
        test_scores[i] = np.mean(nested_score['test_score'])
    return train_scores, test_scores

#taken from starter code ps4
def plot(train_scores, test_scores, clf_strs) :
    """Plot performance."""
    
    labels = ["training", "testing"]
    ind = np.arange(len(labels))    # x locations for groups
    width = 1 / (len(clf_strs) + 1) # width of the bars
    
    # text annotation
    def autolabel(rects) :
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2., height),
                        xytext=(0, 3), textcoords='offset points', # 3 points vertical offset
                        ha='center', va='bottom')
    
    # bar plot with error bars
    fig, ax = plt.subplots()
    for i, clf_str in enumerate(clf_strs) :
        means = (train_scores[clf_str].mean(), test_scores[clf_str].mean())
        stds = (train_scores[clf_str].std(), test_scores[clf_str].std())
        
        rects = ax.bar(ind + width * i, means, width, yerr=stds, label=clf_str)
        autolabel(rects)
    
    # axes
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Nested Cross-Validation Performance')
    ax.set_xticks(ind + width * (len(clf_strs) - 1) / 2.)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)    # lower right
    fig.tight_layout()
    
    plt.show()

def main(args, numtrials): 
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
    y = np.ravel(y)
    train_scores = {}
    test_scores = {}
    clf_strs = ['dummy','lr', 'svc','rf', 'ada']
    for clf_str in clf_strs:
        clf, param_grid = get_classifier(clf_str)
        train, test = get_performance(clf, X, y, numtrials, param_grid)
        train_scores[clf_str] = train
        test_scores[clf_str] = test
        print(f"{clf_str}")
        print(f"\ttraining accuracy: {np.mean(train):.3g} +/- {np.std(train):.3g}")
        print(f"\ttest accuracy:     {np.mean(test):.3g} +/- {np.std(test):.3g}")

    plot(train_scores, test_scores, clf_strs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-generate", action="store_true",
                        help = "Regenerate Data")
    args = parser.parse_args()
    main(args, 10)