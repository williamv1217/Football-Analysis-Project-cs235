from sklearn.linear_model import LinearRegression

def linearReg(training_set_x, training_set_y, test_set_x, test_set_y):
	linReg = LinearRegression(copy_X=True)
	linReg.fit(training_set_x, training_set_y)
	predicted_outcomes = linReg.predict(test_set_x)	
	
	difference = 0.0
	difference_sq = 0.0
	for i in range(len(test_set_y)):
		print(str(predicted_outcomes[i][0]) + ' '+ str(test_set_y[i][0]))
		difference = difference + abs(predicted_outcomes[i][0] - test_set_y[i][0])
		difference_sq = difference_sq + (predicted_outcomes[i][0] - test_set_y[i][0])*(predicted_outcomes[i][0] - test_set_y[i][0])

	print(difference)
	print(difference_sq)
