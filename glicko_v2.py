import math
from functools import partial
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
import time
from utils import time_function
from metrics import accuracy, log_loss
from data_utils import get_dataset, jax_preprocess

@partial(jax.vmap, in_axes=(0, 0, None, None))
@jax.jit
def time_dynamics(rd2, lp, t, c2):
    rd2_new = jax.lax.select(
        jnp.logical_or(lp == t, lp == -1),
        rd2,
        rd2 + (c2 * (t - lp))
    )
    return rd2_new

@jax.jit
def g(rd_squared, three_q2_over_pi2):
    return 1.0 / jnp.sqrt(1.0 + ((rd_squared) * three_q2_over_pi2))

SIGN_FLIP = jnp.array([1.0, -1.0], dtype=jnp.float32)


class Glicko(OnlineRatingSystem):
    def __init__(
        self,
        competitors,
        initial_mu: float = 1500.0,
        initial_rd: float = 350.0,
        c: float = 63.2,
        scale: float = 400.0,
        base: float = 10.0,
    ):
        super().__init__(competitors)
        q = math.log(base) / scale
        q2 = q ** 2.0
        self.params = {
            'initial_mu' : initial_mu,
            'initial_rd2': initial_rd ** 2.0,
            'c2': c ** 2.0,
            'q': q,
            'q2': q2,
            'tq2_pi2': (3.0 * q2) / (math.pi ** 2.0)
        }

    def get_init_state(self, initial_mu, initial_rd2, **kwargs):
        # indices of state are [mu, rd2, last_played]
        # state = jnp.zeros((2, self.num_competitors), dtype=jnp.float32)
        # state = state.at[0].set(initial_mu)
        # state = state.at[1].set(initial_rd2)
        # state = state.at[2].set(-1.0)

        state = {
            'mu' : jnp.full(self.num_competitors, initial_mu),
            'rd2': jnp.full(self.num_competitors, initial_rd2)
        }
        return state


    @staticmethod
    @partial(jax.jit, static_argnums=(4,5,6,7,8))
    def update(c_idxs, time_step, outcome, state, initial_rd2, c2, q, q2, tq2_pi2, **kwargs):
        mu = state['mu']
        rd2 = state['rd2']
        # lp = state[2, c_idxs]

        cur_mu = mu[c_idxs]
        cur_rd2 = rd2[c_idxs]

        # rd2 = time_dynamics(rd2, lp, time_step, c2)
        cur_rd2 = cur_rd2 + c2

        flip_g = jnp.flip(g(cur_rd2, tq2_pi2))
        mu_diffs = (cur_mu * SIGN_FLIP).sum() * SIGN_FLIP
        probs = jax.nn.sigmoid(q * flip_g * mu_diffs)
        d2_inv = q2 * jnp.square(flip_g) * probs * (1.0 - probs)

        both_outcomes = jnp.array([outcome, 1.0 - outcome], dtype=jnp.float32)
        mu_update_num = q * flip_g * (both_outcomes - probs)
        mu_update_denom = (1.0 / cur_rd2) + d2_inv
        mu_updates = mu_update_num / mu_update_denom
        new_rd2s = 1.0 / mu_update_denom

        # state = state.at[0, c_idxs].add(mu_updates)
        # state = state.at[1, c_idxs].set(new_rd2s)
        # state = state.at[2, c_idxs].set(time_step.astype(jnp.float32))

        mu = mu.at[c_idxs].add(mu_updates)
        rd2 = rd2.at[c_idxs].set(new_rd2s)
        prob = (probs[0] + (1.0 - probs[1])) / 2.0

        new_state = {
            'mu' : mu,
            'rd2': rd2
        }
        return new_state, prob

if __name__ == '__main__':
    # dataset = get_dataset("smash_melee", '7D')
    # dataset = get_dataset("league_of_legends", '7D')
    dataset = get_dataset("smash_melee", '7D')



    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)


    glicko = Glicko(dataset.competitors)
    start_time = time.time()
    ratings, probs = glicko.fit(matchups, time_steps, outcomes)
    acc = accuracy(probs, outcomes)
    loss = log_loss(probs, outcomes)
    print(f'acc: {acc:.4f}')
    print(f'log_loss: {loss:.4f}')
    duration = time.time() - start_time
    print(f'duration (s): {duration}')


    # n_samples = 1000
    # rng = jax.random.PRNGKey(0)
    # sweep_params = {
    #     'k': jax.random.uniform(rng, shape=(n_samples,), minval=2.0, maxval=128.0),
    #     'scale': jax.random.uniform(rng, shape=(n_samples,), minval=10.0, maxval=500.0),
    #     'base': jax.random.uniform(rng, shape=(n_samples,), minval=2.0, maxval=20.0),
    # }
    # # sweep_params = {
    # #     'k': jax.random.uniform(rng, shape=(n_samples,), minval=85.0, maxval=100.0),
    # #     'scale': jax.random.uniform(rng, shape=(n_samples,), minval=360.0, maxval=400.0),
    # #     'base': jax.random.uniform(rng, shape=(n_samples,), minval=13.0, maxval=16.0),
    # # }
    # start_time = time.time()
    # all_ratings, all_probs, best_idx = elo.sweep2(matchups, None, outcomes, sweep_params)
    # duration = time.time() - start_time
    # print(f'duration (s): {duration}')

    # mean_probs = all_probs.mean(axis=0)
    # acc = accuracy(mean_probs, outcomes)
    # loss = log_loss(mean_probs, outcomes)
    # print(f'mean acc: {acc:.4f}')
    # print(f'mean log loss: {loss:.4f}')


