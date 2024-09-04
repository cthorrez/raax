import math
import jax
import jax.numpy as jnp
from riix.models.elo import Elo
from functools import partial
from data_utils import gimmie_data

def generate_hyperparam_grid(param_ranges, num_samples, seed=0):
    key = jax.random.PRNGKey(seed)
    grid = {}
    for param, (min_val, max_val) in param_ranges.items():
        key, subkey = jax.random.split(key)
        grid[param] = jax.random.uniform(subkey, (num_samples,), minval=min_val, maxval=max_val)
    
    return grid


def log_loss(probs, outcomes, axis=0):
    return -(outcomes * jnp.log(probs) + (1.0 - outcomes) * jnp.log(1.0 - probs)).mean(axis=axis)

def acc(probs, outcomes, axis=0):
    corr = ((probs > 0.5) == outcomes).astype(jnp.float32).sum(axis=axis) + (0.5 * (probs == 0.5).astype(jnp.float32)).sum(axis=axis)
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
        best_idx = jnp.nanargmax(accuracy)
        # best_idx = jnp.nanargmin(loss)
        return final_vals, final_probs, best_idx

class Elo(RatingSystem):
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

def clayto_loss(locs, scales, outcome):
    z = jnp.log(10.0) / jnp.sqrt(jnp.square(scales).sum())
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
        loc=0.0,
        scale=1.0,
        lr=0.01,
    ):
        super().__init__(num_competitors)
        self.params = {'loc': loc, 'scale': scale, 'lr': lr}

    def initialize(self, loc, scale, **kwargs):
        locs = jnp.full(self.num_competitors, loc)
        scales = jnp.full(self.num_competitors, scale)
        return (locs, scales)

    @staticmethod
    def update_fun(prev_val, x, lr, **kwargs):
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
    game = 'league_of_legends'
    # game = 'starcraft2'
    # game = 'smash_melee'
    # game = 'dota2'
    # game = 'rocket_league'

    matchups, outcomes, num_competitors = gimmie_data(game)
    test_frac = 0.2
    test_idx = int(matchups.shape[0] * (1.0 - test_frac))
    # elo_scale = 1.0
    # elo_k = 0.025
    elo_scale = 400.0
    elo_k = 32.0

    elo = Elo(
        num_competitors=num_competitors,
        loc=0.0,
        scale=elo_scale,
        k=elo_k
    )
    sweep_params = generate_hyperparam_grid({'scale' : (100,1000), 'k' : {1, 200}}, num_samples=2000)
    locs, probs, best_idx = elo.sweep(matchups, outcomes, sweep_params)

    print('locs:', locs[best_idx].min(), locs[best_idx].max())
    print('probs', probs[best_idx].min(), probs[best_idx].mean(), probs[best_idx].max())
    print('log loss', log_loss(probs[best_idx, test_idx:], outcomes[test_idx:]))
    print('acc', acc(probs[best_idx, test_idx:], outcomes[test_idx:]))

    clayto_scale = elo_scale / math.sqrt(2.0)
    clayto_lr = elo_k / (math.log(10.0) / elo_scale)

    clayto = Clayto(
        num_competitors=num_competitors,
        loc=0.0,
        scale=clayto_scale,
        lr=clayto_lr,
    )
    scale_ranges = (100 / math.sqrt(2.0), 1000 / math.sqrt(2))
    lr_ranges = (1 / (math.log(10.0) / 100), 200 / (math.log(10.0) / 1000))
    print(scale_ranges)
    print(lr_ranges)
    
    sweep_params = generate_hyperparam_grid({'scale' : scale_ranges, 'lr' : lr_ranges}, num_samples=2000)
    
    (locs, scales), probs, best_idx = clayto.sweep(matchups, outcomes, sweep_params)
    print('locs:', locs[best_idx].min(), locs[best_idx].max())
    print('scales:', scales[best_idx].min(), scales[best_idx].mean(), scales[best_idx].max())
    print('probs', probs[best_idx].min(), probs[best_idx].mean(), probs[best_idx].max())
    print('log loss', log_loss(probs[best_idx, test_idx:], outcomes[test_idx:]))
    print('acc', acc(probs[best_idx, test_idx:], outcomes[test_idx:]))



if __name__ == '__main__':
    main()

    