from sklearn.neural_network import MLPRegressor
import numpy as np

def neuralNet(training_set_x, training_set_y, test_set_x, test_set_y):
	training_set_x = np.array(training_set_x, dtype = np.float64)
	training_set_y = np.array(training_set_y, dtype = np.float64)
	test_set_x = np.array(test_set_x, dtype = np.float64)
	test_set_y = np.array(test_set_y, dtype = np.float64) 
	
	nnet = MLPRegressor(activation = 'identity', solver='sgd', batch_size = 'auto',
		alpha=1e-5, verbose = True, random_state=1)
	nnet.fit(training_set_x, training_set_y)
	predicted_outcomes = nnet.predict(test_set_x)	

	difference = 0.0
	difference_sq = 0.0
	for i in range(len(test_set_y)):
		print(str(predicted_outcomes[i]) + ' '+ str(test_set_y[i][0]))
		difference = difference + abs(predicted_outcomes[i] - test_set_y[i][0])
		difference_sq = difference_sq + (predicted_outcomes[i] - test_set_y[i][0])*(predicted_outcomes[i] - test_set_y[i][0])

	print(difference)
	print(difference_sq)
