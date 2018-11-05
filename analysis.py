import pandas as pd
import numpy as np

merged_dataset = pd.read_csv('./football_events/merged.csv')

#print merged_dataset.head()

team_rank =['high','med','low']

count_of_games = {}
count_of_foul = {}
count_of_cards = {}

condition_for_card = ((merged_dataset['event_type'] >= 4) & (merged_dataset['event_type']<= 6))

for rank_home in team_rank:
	for rank_away in team_rank:
		condition = merged_dataset['ht_rank'].str.contains(rank_home) & merged_dataset['at_rank'].str.contains(rank_away)
		count_of_games[(rank_home+'-'+rank_away)] = merged_dataset[condition]['id_odsp'].nunique()
		count_of_foul[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & (merged_dataset['event_type'] == 3)])
		count_of_cards[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & condition_for_card])

print(count_of_games);
print(count_of_foul);
print(count_of_cards);

for rank_home in team_rank:
	for rank_away in team_rank:
		games = count_of_games[(rank_home+'-'+rank_away)];
		fouls = count_of_foul[(rank_home+'-'+rank_away)];
		cards = count_of_cards[(rank_home+'-'+rank_away)]
		print ((rank_home+'-'+rank_away) +' games '+ repr(games) + ' fouls '+ repr(fouls) + ' fouls/game '+ repr(fouls/games) +' cards '+ repr(cards) + ' cards/game '+ repr(cards/games));
#condtion_low_high = (merged_dataset['ht_rank'] == 'low' & merged_dataset['at_rank'] == 'high')

#dataset_condition_low_high = merged_dataset[condtion_low_high]; dataset_condition_low_high.head()

#print dataset_condition_low_high.head()
