people = {'shreyas':1,'kt':2, 'unknown': 3}

def getNameFromValue(val):
	res = people.keys()[people.values().index(int(val))]
	return res