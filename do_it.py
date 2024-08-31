import time
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset
from riix.models.elo import Elo
from functools import partial
from data_utils import gimmie_data

def log_loss(probs, outcomes):
    return (outcomes * jnp.log(probs) + (1.0 - outcomes) * jnp.log(1.0 - probs)).mean()

def acc(probs, outcomes):
    corr = ((probs > 0.5) == outcomes).astype(jnp.float32).sum() + (0.5 * (probs == 0.5).astype(jnp.float32)).sum()
    return corr / outcomes.shape[0]

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


def clayto_loss(locs, scales, outcome):
    z = jnp.sqrt(jnp.square(scales).sum())
    logit = z * (locs * jnp.array([1.0, -1.0])).sum()
    prob = jax.nn.sigmoid(logit)
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss, prob

clayto_grad = jax.grad(
    fun=clayto_loss,
    argnums=(0,1),
    has_aux=True
)

class Clayto(RatingSystem):
    def __init__(
        self,
        num_competitors,
        loc = 0.0,
        scale = 1.0,
        lr = 0.01,
    ):
        super().__init__(num_competitors)
        locs = jnp.zeros(num_competitors) + loc
        scales = jnp.zeros(num_competitors) + scale
        self.init_val = (locs, scales)
        self.update_fun = partial(self._update_fun, lr=lr)


    @staticmethod
    def loss(locs, scales, outcome):
        z = jnp.sqrt(jnp.square(scales).sum())
        logit = z * locs * jnp.array([1.0, -1.0])
        prob = jax.nn.sigmoid(logit)
        loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
        return loss, prob
    
    @staticmethod
    def _update_fun(prev_val, x, lr):
        competitors = x['matchups']
        outcome = x['outcomes']
        locs, scales = prev_val
        c_locs = locs[competitors]  
        c_scales = scales[competitors]
        grad, prob = clayto_grad(c_locs, c_scales, outcome)
        new_locs = locs.at[competitors].add(lr * grad[0])
        new_scales = scales.at[competitors].add(lr * grad[1])
  
        return (new_locs, new_scales), prob
    

        
def main():
    matchups, outcomes = gimmie_data('league_of_legends')
    num_competitors = jnp.unique(matchups).max()
    elo = Elo(num_competitors=num_competitors)
    ratings, probs = elo.run(matchups, outcomes)
    print('ratings', ratings.min(), ratings.max())
    print('probs', probs.min(), probs.mean(), probs.max())
    print('log loss', log_loss(probs[-10000:], outcomes[-10000:]))
    print('acc', acc(probs[-10000:], outcomes[-10000:]))



    clayto = Clayto(
        num_competitors=num_competitors,
        loc=0.0,
        scale=1.0,
        lr=0.2,
    )
    (locs, scales), probs = clayto.run(matchups, outcomes)
    print('locs:', locs.min(), locs.max())
    print('scales:', scales.min(), scales.mean(), scales.max())
    print('probs', probs.min(), probs.mean(), probs.max())
    print('log loss', log_loss(probs[-10000:], outcomes[-10000:]))
    print('acc', acc(probs[-10000:], outcomes[-10000:]))



if __name__ == '__main__':
    main()

    