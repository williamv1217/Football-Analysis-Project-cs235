#from sets import Set
import pickle

# sakib code to get features from a dataset and save into a pickle file

# eventType
ATTEMPT = 1
CORNER = 2
FOUL = 3
YELLOW = 4
YELLOW2 = 5
RED = 6
SUBSTITUTION = 7
FREE_KICK = 8
OFFSIDE = 9
HAND_BALL = 10
PENALTY = 11

# eventType2
KEY_PASS = 1
FAILED_THROUGH_PASS = 2

# shotOutcome
ON_TARGET = 1
OFF_TARGET = 2
BLOCKED = 3
HIT_THE_BAR = 4

# print('merged datasets')
# merged_dataset = pd.read_csv('../football_events/merged.csv')
#
# team_rank = ['high', 'med', 'low']
#
# # idx = list(range(50,len(merged_dataset),500))
# # merged_dataset = merged_dataset.iloc[idx]
#
# print('teams set()')
# teams = set()
#
# print('home away unique()')
# home_teams = merged_dataset['ht'].unique()
# away_teams = merged_dataset['at'].unique()
#
# print('update teams()')
# teams.update(home_teams)
# teams.update(away_teams)
#
# print('merged date time')
# # changing date from object/string to date
# merged_dataset['date'] = pd.to_datetime(merged_dataset['date'])
#
# print('create train and test times')
# training_dataset = merged_dataset[merged_dataset['date'] <= '31-DEC-2015']
# test_dataset = merged_dataset[merged_dataset['date'] > '31-DEC-2015']

#print(pickle.load(open('set_y.pkl', 'rb')))
tr_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
tr_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
te_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
te_set_y = pickle.load(open('test_set_y.pkl', 'rb'))


print('create train sets')
training_set_x, training_set_y = tr_set_x, tr_set_y #datasets(training_dataset)
# print(training_set_x)
# print(training_set_y)

print('create test sets')
test_set_x, test_set_y = te_set_x, te_set_y #datasets(test_dataset)
# print(test_set_x)
# print(test_set_y)

print('doing knn')
# knn(training_set_x, training_set_y, test_set_x, test_set_y)


#neuralNet(training_set_x, training_set_y, test_set_x, test_set_y)
#linearReg(training_set_x, training_set_y, test_set_x, test_set_y)