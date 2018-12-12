from prediction.neural_network import neural_network
from prediction.knn import knn
from prediction.linear_regression import linear_regression
import numpy as np



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


knn(np.array([1,1,1,14,5,6,3,0,0,17,2,1,0,0,0,6,3,5,0]), np.array([2,1,1,16,7,17,2,0,0,6,3,0,0,0,0,5,6,5,0]), 'chelsea', 'man city')
linear_regression(np.array([1,1,1,14,5,6,3,0,0,17,2,1,0,0,0,6,3,5,0]), np.array([2,1,1,16,7,17,2,0,0,6,3,0,0,0,0,5,6,5,0]), 'chelsea', 'man city')