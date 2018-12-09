from sklearn.neighbors import KNeighborsRegressor

def knn(training_set_x, training_set_y, test_set_x, test_set_y):

	neigh = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto')
	neigh.fit(training_set_x, training_set_y)

	predicted_outcomes = neigh.predict(test_set_x)

	difference = 0.0
	difference_sq = 0.0
	for i in range(len(test_set_y)):
		print(str(predicted_outcomes[i][0]) + ' '+ str(test_set_y[i][0]))
		difference = difference + abs(predicted_outcomes[i][0] - test_set_y[i][0])
		difference_sq = difference_sq + (predicted_outcomes[i][0] - test_set_y[i][0])*(predicted_outcomes[i][0] - test_set_y[i][0])

	print(difference)
	print(difference_sq)
