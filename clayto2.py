import math
from functools import partial
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
from utils import time_function
from metrics import log_loss, accuracy
from data_utils import get_dataset, jax_preprocess

@jax.jit
def loss(m_a, m_b, v_a, v_b, alpha, outcome):
    scale = jnp.sqrt(v_a ** 2.0 + v_b ** 2.0)
    prob = jax.nn.sigmoid((alpha / scale) * (m_a - m_b))
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss, prob

grad_fn = jax.jit(jax.grad(
    fun=loss,
    argnums=(0,1,2,3),
    has_aux=True
))

class Clayto(OnlineRatingSystem):
    def __init__(
        self,
        competitors,
        initial_loc: float = 1500.0,
        initial_scale: float = 200,
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
        loc = jnp.full(shape=(self.num_competitors,), fill_value=initial_loc, dtype=jnp.float32)
        scale = jnp.full(shape=(self.num_competitors,), fill_value=initial_scale, dtype=jnp.float32)
        return (loc, scale)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, idx_a, idx_b, time_step, outcome, state, loc_lr, scale_lr, alpha, **kwargs):
        loc, scale = state
        m_a, m_b = loc[idx_a], loc[idx_b]
        v_a, v_b = scale[idx_a], scale[idx_b]
        grad, prob = grad_fn(m_a, m_b, v_a, v_b, alpha, outcome)
        loc = loc.at[idx_a].add(loc_lr * grad[0])
        loc = loc.at[idx_b].add(loc_lr * grad[1])
        scale = scale.at[idx_a].add(scale_lr * grad[2])
        scale = scale.at[idx_b].add(scale_lr * grad[3])
        new_state = (loc, scale)
        return new_state, prob

if __name__ == '__main__':
    dataset = get_dataset("league_of_legends", '7D')

    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)

    clayto = Clayto(dataset.competitors)
    (loc, scale), probs = clayto.fit(matchups, None, outcomes)
    print(loc.min(), loc.mean(), loc.max())
    print(scale.min(), scale.mean(), scale.max())
    print(probs.min(), probs.mean(), probs.max())
    acc = accuracy(probs, outcomes)
    loss = log_loss(probs, outcomes)
    print(f'acc: {acc:.4f}')
    print(f'log loss: {loss:.4f}')


    n_samples = 500
    rng = jax.random.PRNGKey(0)
    sweep_params = {
        'init_scale': jax.random.uniform(rng, shape=(n_samples,), minval=450.0, maxval=550.0),
        'loc_lr': jax.random.uniform(rng, shape=(n_samples,), minval=45000.0, maxval=60000.0),
        'scale_lr': jax.random.uniform(rng, shape=(n_samples,), minval=9000.0, maxval=13000.0),
        'scale': jax.random.uniform(rng, shape=(n_samples,), minval=1.0, maxval=1.0),
        'base': jax.random.uniform(rng, shape=(n_samples,), minval=math.e, maxval=math.e),
    }
    all_ratings, all_probs, best_idx = clayto.sweep(matchups, None, outcomes, sweep_params)

    mean_probs = all_probs.mean(axis=0)
    acc = accuracy(mean_probs, outcomes)
    loss = log_loss(mean_probs, outcomes)
    print(f'mean acc: {acc:.4f}')
    print(f'mean log loss: {loss:.4f}')

