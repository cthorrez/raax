//
func Cache add(double[] ratings, oldRatings, row) {
	
	ua, ub = elo(oldRatings, row)
	ratings[a_id] += ua
	ratings[a_id] += ua
	
	return cache
}
commit, staged
// every period. 1 exec, cache=Space(nPlayer)
func commit(oldratings, ratings) {

	oldratings = Ratings
}

func sunnieRaax(rows[] dataset, int nPlayers ) {
	// var cache = double[nPlayers] (100)
	var ratings = double[nPlayers]  (100)
	var curStage = 0;
	var oldRatings = ratings
	for (int i =0; i <= dataset.size, i++) {
		if dataset[i].isNewPeriod {
			commit(cache, &oldRatings, &ratings)
			reset(cache)
		}
		
		add(oldratings, ratings, &cache, dataset[i])
	}

	commit(cache, &oldRatings, &ratings)
	return ratings
}