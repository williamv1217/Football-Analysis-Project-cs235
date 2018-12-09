import pandas as pd
import numpy as np
from sets import Set
from dataset import dataset
from knn import knn
from neural import neuralNet
from sklearn.neural_network import MLPRegressor

			
#eventType
ATTEMPT 		= 1;
CORNER 			= 2;
FOUL 			= 3;
YELLOW 			= 4;
YELLOW2 		= 5;
RED 			= 6;
SUBSTITUTION	= 7;
FREE_KICK 	 	= 8;
OFFSIDE 	 	= 9;
HAND_BALL 	 	= 10;
PENALTY 		= 11;
	
	
#eventType2
KEY_PASS 			= 1;
FAILED_THROUGH_PASS = 2;
	
	
#shotOutcome
ON_TARGET 	= 1;
OFF_TARGET 	= 2;
BLOCKED 	= 3;
HIT_THE_BAR = 4;

merged_dataset = pd.read_csv('./football_events/merged.csv')

team_rank =['high','med','low']

#idx = list(range(50,len(merged_dataset),500))
#merged_dataset = merged_dataset.iloc[idx];

teams = set()

home_teams = merged_dataset['ht'].unique();
away_teams = merged_dataset['at'].unique();

teams.update(home_teams)
teams.update(away_teams)

#changing date from object/string to date
merged_dataset['date'] = pd.to_datetime(merged_dataset['date'])

training_dataset = merged_dataset[merged_dataset['date'] <= '31-DEC-2015']
test_dataset = merged_dataset[merged_dataset['date'] > '31-DEC-2015']

training_set_x, training_set_y = dataset(training_dataset)
#print(training_set_x)
#print(training_set_y)

test_set_x, test_set_y = dataset(test_dataset)
#print(test_set_x)
#print(test_set_y)

#knn(training_set_x, training_set_y, test_set_x, test_set_y)

neuralNet(training_set_x, training_set_y, test_set_x, test_set_y)




