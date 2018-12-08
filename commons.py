team_rank =['high','med','low']

def rankToRank(string):
	i = 0
	for x in team_rank:
		i = i +1 ;
		if string == x:
			return i
			
