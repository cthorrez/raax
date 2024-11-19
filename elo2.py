import math
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
import time
from utils import time_function
from metrics import accuracy, log_loss
from data_utils import get_dataset, jax_preprocess

class Elo(OnlineRatingSystem):
    def __init__(
        self,
        competitors,
        initial_rating: float = 1500,
        k: float = 32.0,
        scale: float = 400.0,
        base: float = 10.0,
    ):
        super().__init__(competitors)
        self.params = {
            'initial_rating' : initial_rating,
            'k': k,
            'alpha': math.log(base) / scale
        }

    def get_init_state(self, initial_rating, **kwargs):
        return jnp.full(shape=(self.num_competitors,), fill_value=initial_rating, dtype=jnp.float32)

    def update(self, c_idxs, time_step, outcome, state, k, alpha, **kwargs):
        r_a, r_b = state[c_idxs]
        prob = jax.nn.sigmoid(alpha * (r_a - r_b))
        update = k * (outcome - prob)
        state = state.at[c_idxs[0]].add(update)
        state = state.at[c_idxs[1]].add(-update)
        return state, prob

if __name__ == '__main__':
    # dataset = get_dataset("smash_melee", '7D')
    # dataset = get_dataset("league_of_legends", '7D')
    dataset = get_dataset("smash_melee", '7D')



    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)

    start_time = time.time()
    elo = Elo(dataset.competitors)
    ratings, probs = elo.fit(matchups, None, outcomes)
    acc = ((probs >= 0.5) == outcomes).mean()
    duration = time.time() - start_time
    print(f'duration (s): {duration}')
    print(f'acc: {acc:.4f}')

    n_samples = 100
    rng = jax.random.PRNGKey(0)
    sweep_params = {
        'k': jax.random.uniform(rng, shape=(n_samples,), minval=2.0, maxval=128.0),
        'scale': jax.random.uniform(rng, shape=(n_samples,), minval=10.0, maxval=500.0),
        'base': jax.random.uniform(rng, shape=(n_samples,), minval=2.0, maxval=20.0),
    }
    # sweep_params = {
    #     'k': jax.random.uniform(rng, shape=(n_samples,), minval=85.0, maxval=100.0),
    #     'scale': jax.random.uniform(rng, shape=(n_samples,), minval=360.0, maxval=400.0),
    #     'base': jax.random.uniform(rng, shape=(n_samples,), minval=13.0, maxval=16.0),
    # }
    start_time = time.time()
    all_ratings, all_probs, best_idx = elo.sweep(matchups, None, outcomes, sweep_params)
    # all_ratings, all_probs, best_idx = elo.sweep2(matchups, None, outcomes, sweep_params)
    duration = time.time() - start_time
    print(f'duration (s): {duration}')

    # mean_probs = all_probs.mean(axis=0)
    # acc = accuracy(mean_probs, outcomes)
    # loss = log_loss(mean_probs, outcomes)
    # print(f'mean acc: {acc:.4f}')
    # print(f'mean log loss: {loss:.4f}')


