import pandas as pd
import numpy as np

merged_dataset = pd.read_csv('./football_events/merged.csv')

#print merged_dataset.head()

team_rank =['high','med','low']

count_of_games = {}
count_of_foul = {}
count_of_cards = {}
count_of_foul_home_team = {}
count_of_cards_home_team = {}

condition_for_card = ((merged_dataset['event_type'] >= 4) & (merged_dataset['event_type']<= 6))
condition_for_home_team_action = (merged_dataset['side'] ==1)
condition_for_foul = (merged_dataset['event_type'] == 3)

for rank_home in team_rank:
	for rank_away in team_rank:
		condition = merged_dataset['ht_rank'].str.match(rank_home) & merged_dataset['at_rank'].str.match(rank_away)
		count_of_games[(rank_home+'-'+rank_away)] = merged_dataset[condition]['id_odsp'].nunique()
		count_of_foul[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & condition_for_foul])
		count_of_cards[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & condition_for_card])
		count_of_foul_home_team[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & condition_for_foul & condition_for_home_team_action])
		count_of_cards_home_team[(rank_home+'-'+rank_away)] = len(merged_dataset[condition & condition_for_card & condition_for_home_team_action])

print(count_of_games);
print(count_of_foul);
print(count_of_cards);

for rank_home in team_rank:
	for rank_away in team_rank:
		games = count_of_games[(rank_home+'-'+rank_away)];
		fouls = count_of_foul[(rank_home+'-'+rank_away)];
		fouls_by_home_team = count_of_foul_home_team[(rank_home+'-'+rank_away)];
		cards = count_of_cards[(rank_home+'-'+rank_away)];
		cards_by_home_team = count_of_cards_home_team[(rank_home+'-'+rank_away)];
		
		print ('\n' + rank_home+'-'+rank_away + ' games '+ repr(games));
		print ('fouls '+ repr(fouls) + ' fouls/game '+ repr(fouls/games)); #+ 
		print ('fouls by home team/game ' + repr(fouls_by_home_team/games) + ' fouls by away team/game ' + repr((fouls - fouls_by_home_team)/games));
		print ('cards '+ repr(cards) + ' cards/game '+ repr(cards/games));
		print ('cards by home team/game ' + repr(cards_by_home_team/games) + ' cards by away team/game ' + repr((cards - cards_by_home_team)/games));
#condtion_low_high = (merged_dataset['ht_rank'] == 'low' & merged_dataset['at_rank'] == 'high')

#dataset_condition_low_high = merged_dataset[condtion_low_high]; dataset_condition_low_high.head()

#print dataset_condition_low_high.head()
