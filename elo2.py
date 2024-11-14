import math
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
from utils import time_function
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
        self.initial_rating = initial_rating
        self.alpha = math.log(base) / scale
        self.k = k
    
    def initialize_state(self):
        return jnp.full(shape=(self.num_competitors,), fill_value=self.initial_rating, dtype=jnp.float32)

    def update(self, idx_a, idx_b, time_step, outcome, state):
        r_a = state[idx_a]
        r_b = state[idx_b]
        prob = jax.nn.sigmoid(self.alpha * (r_a - r_b))
        update = self.k * (outcome - prob)
        state = state.at[idx_a].add(update)
        state = state.at[idx_b].add(-update)
        return state, prob

if __name__ == '__main__':
    dataset = get_dataset("league_of_legends", '7D')

    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)

    elo = Elo(dataset.competitors)
    ratings, probs = elo.fit(matchups, None, outcomes)
    print(ratings, probs)
    acc = ((probs >= 0.5) == outcomes).mean()
    print(f'acc: {acc:.4f}')