import time
import math
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import numpy as np
from riix.models.elo import Elo as riix_Elo
from functools import partial
from data_utils import load_dataset

jax.config.update("jax_enable_x64", True)

@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name} duration (s): {end_time - start_time:.4f}")

def generate_hyperparam_grid(param_ranges, num_samples, seed=0):
    key = jax.random.PRNGKey(seed)
    grid = {}
    for param, (min_val, max_val) in param_ranges.items():
        key, subkey = jax.random.split(key)
        grid[param] = jax.random.uniform(subkey, (num_samples,), minval=min_val, maxval=max_val, dtype=jnp.float64)
    
    return grid


def log_loss(probs, outcomes, axis=0):
    return -(outcomes * jnp.log(probs) + (1.0 - outcomes) * jnp.log(1.0 - probs)).mean(axis=axis)

def acc(probs, outcomes, axis=0):
    corr = 0.0
    corr += ((probs > 0.5) & (outcomes == 1.0)).astype(jnp.float32).sum(axis=axis)
    corr += ((probs < 0.5) & (outcomes == 0.0)).astype(jnp.float32).sum(axis=axis)
    corr += (0.5 * (probs == 0.5).astype(jnp.float32)).sum(axis=axis)
    return corr / outcomes.shape[axis]

class RatingSystem:
    def __init__(self, num_competitors: int):
        self.num_competitors = num_competitors

    @staticmethod
    def update_fun(carry, x):
        raise NotImplementedError

    def initialize(self, **params):
        raise NotImplementedError

    def run(self, matchups, outcomes):
        init_val = self.initialize(**self.params)
        return self._run(matchups, outcomes, init_val, **self.params)

    def _run(self, matchups, outcomes, init_val, **params):
        update_fun = partial(self.update_fun, **params)
        final_val, probs = jax.lax.scan(
            f=update_fun,
            init=init_val,
            xs={'matchups': matchups, 'outcomes': outcomes},
        )
        return final_val, probs

    def sweep(self, matchups, outcomes, sweep_params):
        start_time = time.time()
        fixed_params = {k: v for k, v in self.params.items() if k not in sweep_params}

        def run_single(matchups, outcomes, sweep_params):
            all_params = {**fixed_params, **sweep_params}
            init_val = self.initialize(**all_params)
            return self._run(matchups, outcomes, init_val, **all_params)

        in_axes = (None, None, {param: 0 for param in sweep_params})
        run_many = jax.vmap(run_single, in_axes=in_axes)

        final_vals, final_probs = run_many(matchups, outcomes, sweep_params)
        loss = log_loss(final_probs, jnp.expand_dims(outcomes, 0), axis=1)
        accuracy = acc(final_probs, jnp.expand_dims(outcomes, 0), axis=1)
        # best_idx = jnp.nanargmax(accuracy)
        best_idx = jnp.nanargmin(loss)
        duration = time.time() - start_time
        print(f'duration (s): {duration:0.4f}')
        for param, vals in sweep_params.items():
            print(f'best {param}: {vals[best_idx]}')
        return final_vals, final_probs, best_idx

class OnlineElo(RatingSystem):
    def __init__(
        self,
        num_competitors,
        loc=1500.0,
        scale=400.0,
        k=32.0
    ):
        super().__init__(num_competitors)
        self.params = {'loc': loc, 'scale': scale, 'k': k}

    def initialize(self, loc, **kwargs):
        return jnp.full(self.num_competitors, loc)

    @staticmethod
    def update_fun(ratings, x, scale, k, **kwargs):
        competitors = x['matchups']
        outcome = x['outcomes']
        logit = (jnp.log(10.0) / scale) * (ratings[competitors] * jnp.array([1.0, -1.0])).sum()
        prob = jax.nn.sigmoid(logit)
        update = k * (outcome - prob)
        new_ratings = ratings.at[competitors[0]].add(update)
        new_ratings = new_ratings.at[competitors[1]].add(-update)
        return new_ratings, prob
    
class BatchedElo(RatingSystem):
    def __init__(
        self,
        num_competitors,
        loc=1500.0,
        scale=400.0,
        k=32.0
    ):
        super().__init__(num_competitors)
        self.params = {'loc': loc, 'scale': scale, 'k': k}

    def initialize(self, loc, **kwargs):
        return jnp.full(self.num_competitors, loc)

    @staticmethod
    def update_fun(ratings, x, scale, k, **kwargs):
        competitors = x['matchups']
        outcome = x['outcomes']
        logit = (jnp.log(10.0) / scale) * (ratings[competitors] * jnp.array([1.0, -1.0])).sum()
        prob = jax.nn.sigmoid(logit)
        update = k * (outcome - prob)
        new_ratings = ratings.at[competitors[0]].add(update)
        new_ratings = new_ratings.at[competitors[1]].add(-update)
        return new_ratings, prob
    
def main():
    dataset = load_dataset("league_of_legends")
    dataset = load_dataset("smash_melee")

    np_matchups, np_outcomes, competitors = dataset.matchups, dataset.outcomes, dataset.competitors
    jnp_matchups, jnp_outcomes = jnp.array(np_matchups), jnp.array(np_outcomes)
    riix_elo = riix_Elo(competitors)
    raax_elo = OnlineElo(len(competitors))

    with timer('riix elo'):
        riix_probs = riix_elo.fit_dataset(dataset, return_pre_match_probs=True)
    with timer('raax online elo'):
        raax_probs = raax_elo.run(jnp_matchups, jnp_outcomes)

if __name__ == '__main__':
    main()