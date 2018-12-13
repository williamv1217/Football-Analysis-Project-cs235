team_rank =['high','med','low']

# sakib code for team ranks
def rankToRank(string):
	i = 0
	for x in team_rank:
		i = i +1 ;
		if string == x:
			return i