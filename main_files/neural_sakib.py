from sklearn.neural_network import MLPRegressor
import pickle
import numpy as np
from main_files.math_functions import root_mean_squares
from sklearn.metrics import r2_score, explained_variance_score


def neuralNet(training_set_x, training_set_y, test_set_x, test_set_y):
	nnet = MLPRegressor(activation = 'identity', solver='sgd', batch_size = 'auto',
		alpha=1e-5, verbose = True, hidden_layer_sizes=(5, 2), random_state=0)

	nnet.fit(training_set_x, training_set_y)
	predicted_outcomes = nnet.predict(test_set_x)

	print()
	print('------------------------------')
	print()
	print('root mean square: ', root_mean_squares(test_set_y, predicted_outcomes))
	print('r2_score: ', r2_score(test_set_y, predicted_outcomes))
	print('explained variance: ', explained_variance_score(test_set_y, predicted_outcomes))

tr_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
tr_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
te_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
te_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

training_set_x, training_set_y = np.array(tr_set_x), np.array(tr_set_y)
test_set_x, test_set_y = np.array(te_set_x), np.array(te_set_y)

neuralNet(training_set_x, training_set_y, test_set_x, test_set_y)

x = MLPRegressor()
x.fit(training_set_x, training_set_y)

team1_home =[[1,1,2,13,5,15,1,0,0,19,2,0,0,0,0,6,3,4,0]
,[2,1,2,10,4,19,3,0,1,15,2,0,0,0,0,4,3,3,0]
,[1,1,1,7,3,13,3,0,0,10,2,0,0,0,0,2,1,3,1]
,[2,1,1,15,7,10,1,0,0,13,3,0,0,0,0,2,7,5,1]
,[1,2,1,16,10,6,0,0,0,8,1,0,0,0,0,5,6,5,0]
,[2,2,1,17,10,8,3,0,0,6,1,0,0,0,0,7,6,4,0]
,[1,1,1,14,5,6,3,0,0,17,2,1,0,0,0,6,3,5,0]
,[2,1,1,16,7,17,2,0,0,6,3,0,0,0,0,5,6,5,0]]

team1_away = [[1,1,1,19,4,12,3,0,0,13,0,0,0,0,0,7,10,2,0]
,[2,1,1,9,7,13,6,1,0,12,2,0,0,0,0,4,3,2,0]
,[1,1,2,12,5,16,4,0,0,6,2,0,0,0,0,2,3,5,2]
,[2,1,2,8,3,5,2,0,0,16,0,0,0,0,0,3,5,0,0]]


team1_home= np.array(team1_home)
team1_away = np.array(team1_away)

team1 = []
team2 = []

for row in range(len(team1_home)):
	if team1_home[row][0] == 1:
		team1.append(team1_home[row])
	else:
		team2.append(team1_home[row])

for row in range(len(team1_away)):
	if team1_away[row][0] == 1:
		team2.append(team1_away[row])
	else:
		team1.append(team1_away[row])

team1 = np.array(team1)
team2 = np.array(team2)

team1 = np.average(team1, axis=0)
team2 = np.average(team2, axis=0)

x.predict(team1.reshape(-1, 1))