import math
import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from rating_system import OnlineRatingSystem
from utils import time_function
from data_utils import get_dataset, jax_preprocess


def loss(m_a, m_b, v_a, v_b, alpha, outcome):
    scale = jnp.sqrt(v_a ** 2.0 + v_b ** 2.0)
    prob = jax.nn.sigmoid((alpha / scale) * (m_a - m_b))
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss, prob

grad_fn = jax.grad(
    fun=loss,
    argnums=(0,1,2,3),
    has_aux=True
)

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
        self.initial_loc = initial_loc
        self.initial_scale = initial_scale
        self.loc_lr = loc_lr
        self.scale_lr = scale_lr
        self.alpha = math.log(base) / scale

        
    def initialize_state(self):
        loc = jnp.full(shape=(self.num_competitors,), fill_value=self.initial_loc, dtype=jnp.float32)
        scale = jnp.full(shape=(self.num_competitors,), fill_value=self.initial_scale, dtype=jnp.float32)
        return (loc, scale)

    def update(self, idx_a, idx_b, time_step, outcome, state):
        loc, scale = state
        m_a, m_b = loc[idx_a], loc[idx_b]
        v_a, v_b = scale[idx_a], scale[idx_b]
        grad, prob = grad_fn(m_a, m_b, v_a, v_b, self.alpha, outcome)
        # jax.debug.print('grad: {}', grad)
        loc = loc.at[idx_a].add(self.loc_lr * grad[0])
        loc = loc.at[idx_b].add(self.loc_lr * grad[1])
        scale = scale.at[idx_a].add(self.scale_lr * grad[2])
        scale = scale.at[idx_b].add(self.scale_lr * grad[3])
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
    print((loc != 1500.0).sum())
    acc = ((probs >= 0.5) == outcomes).mean()
    print(f'acc: {acc:.4f}')