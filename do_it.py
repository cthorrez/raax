import time
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from riix.models.elo import Elo
from functools import partial
from data_utils import gimmie_data

class RatingSystem:
    init_val: None

    def __init__(self, num_competitors:int):
        self.num_competitors = num_competitors

    @staticmethod
    def update_fun(carry, x):
        raise NotImplementedError

    def run(self, matchups, outcomes):
        final_val, probs = jax.lax.scan(
            f=self.update_fun,
            init=self.init_val,
            xs={'matchups': matchups, 'outcomes': outcomes},
        )
        return final_val, probs

class Elo(RatingSystem):
    def __init__(
        self,
        num_competitors,
        loc = 1500.0,
        scale = 400.0,
        k = 32.0
    ):
        super().__init__(num_competitors)
        ratings = jnp.zeros(self.num_competitors) + loc
        self.scale = scale
        self.init_val = ratings
        self.k = k
        self.update_fun = partial(self._update_fun, scale=self.scale, k=self.k)

    @staticmethod
    def _update_fun(ratings, x, scale, k):
        competitors = x['matchups']
        outcome = x['outcomes']
        logit = (jnp.log(10.0)/scale) * (ratings[competitors] * jnp.array([1.0, -1.0])).sum()
        prob = jax.nn.sigmoid(logit)
        update = k * (outcome - prob)
        new_ratings = ratings.at[competitors[0]].add(update)
        new_ratings = new_ratings.at[competitors[1]].add(-update)
        return new_ratings, prob
        
def main():
    matchups, outcomes = gimmie_data('league_of_legends')
    num_competitors = jnp.unique(matchups).max()
    elo = Elo(num_competitors=num_competitors)
    ratings, probs = elo.run(matchups, outcomes)
    print(ratings.min(), ratings.max())
    print(probs)

if __name__ == '__main__':
    main()

    