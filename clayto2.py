import math
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
import time
from utils import time_function
from metrics import log_loss, accuracy
from data_utils import get_dataset, jax_preprocess

@jax.jit
def loss(ms, vs, alpha, outcome):
    scale = jnp.sqrt(jnp.square(vs).sum())
    logit = (alpha * ms * jnp.array([1.0, -1.0])).sum() / scale
    prob = jax.nn.sigmoid(logit)
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss, prob

grad_fn = jax.jit(jax.grad(
    fun=loss,
    argnums=(0,1),
    has_aux=True
))

class Clayto(OnlineRatingSystem):
    def __init__(
        self,
        competitors,
        initial_loc: float = 1500.0,
        initial_scale: float = 200.0,
        loc_lr: float = 24000.0,
        scale_lr: float = 1024.0,
        scale: float = 1.0,
        base: float = math.e,
    ):
        super().__init__(competitors)
        self.params = {
            'initial_loc': initial_loc,
            'initial_scale': initial_scale,
            'loc_lr': loc_lr,
            'scale_lr': scale_lr,
            'alpha': math.log(base) / scale
        }

    @partial(jax.jit, static_argnums=(0,))
    def get_init_state(self, initial_loc, initial_scale, **kwargs):
        # Combine loc and scale into a single array
        state = jnp.zeros((2, self.num_competitors), dtype=jnp.float32)
        state = state.at[0].set(initial_loc)  # loc
        state = state.at[1].set(initial_scale)  # scale
        return state

    @staticmethod
    @jax.jit
    def update(c_idxs, time_step, outcome, state, loc_lr, scale_lr, alpha, **kwargs):
        # Extract loc and scale from state
        loc = state[0, c_idxs]
        scale = state[1, c_idxs]
        
        (loc_grad, scale_grad), prob = grad_fn(loc, scale, alpha, outcome)

        # Update indices for both loc and scale at once
        updates = jnp.stack([
            loc_grad * loc_lr,
            scale_grad * scale_lr
        ])
        
        # Single scatter operation to update both loc and scale
        new_state = state.at[:, c_idxs].add(updates)
        
        return new_state, prob


if __name__ == '__main__':
    # dataset = get_dataset("league_of_legends", '7D')
    # dataset = get_dataset("call_of_duty", '7D')
    dataset = get_dataset("smash_melee", '7D')

    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)
    start_time = time.time()
    clayto = Clayto(dataset.competitors)
    (loc, scale), probs = clayto.fit(matchups, None, outcomes)
    print(loc.min(), loc.mean(), loc.max())
    print(scale.min(), scale.mean(), scale.max())
    print(probs.min(), probs.mean(), probs.max())
    acc = accuracy(probs, outcomes)
    loss_val = log_loss(probs, outcomes)
    duration = time.time() - start_time
    print(f'duration (s): {duration}')
    print(f'acc: {acc:.4f}')
    print(f'log loss: {loss_val:.4f}')


    n_samples = 100
    rng = jax.random.PRNGKey(0)
    sweep_params = {
        'init_scale': jax.random.uniform(rng, shape=(n_samples,), minval=600, maxval=1000.0),
        'loc_lr': jax.random.uniform(rng, shape=(n_samples,), minval=36000, maxval=64000),
        'scale_lr': jax.random.uniform(rng, shape=(n_samples,), minval=2048.0, maxval=8296.0),
    }
    start_time = time.time()
    all_ratings, all_probs, best_idx = clayto.sweep2(matchups, None, outcomes, sweep_params)

    duration = time.time() - start_time
    print(f'duration (s): {duration}')

    # mean_probs = all_probs.mean(axis=0)
    # acc = accuracy(mean_probs, outcomes)
    # loss = log_loss(mean_probs, outcomes)
    # print(f'mean acc: {acc:.4f}')
    # print(f'mean log loss: {loss:.4f}')

