import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import math
from functools import partial
import numpy as np
import jax
jax.default_device = jax.devices("cpu")[0]
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from utils import time_function
from data_utils import get_dataset, jax_preprocess
from riix.utils.data_utils import MatchupDataset
from riix.models.elo import Elo



def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def elo_loss(ratings, outcome):
    logit = (ratings * jnp.array([1.0, -1.0])).sum()
    prob = sigmoid(logit)
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss

elo_grad = jax.grad(elo_loss, argnums=0)

@partial(jax.jit, static_argnums=(2,))
def elo_grad_hard(ratings, outcome, alpha):
    prob = jax.nn.sigmoid(alpha*(ratings[0] - ratings[1]))
    grad = outcome - prob
    return grad

vec_elo_grad_hard = jax.jit(jax.vmap(elo_grad_hard))


def online_elo_update(idx, prev_val, alpha, k):
    ratings = prev_val['ratings']
    comp_idxs = prev_val['matchups'][idx]
    comp_ratings = ratings[comp_idxs]
    outcome = prev_val['outcomes'][idx]
    grad = elo_grad_hard(comp_ratings, outcome, alpha)
    # grad = elo_grad(comp_ratings, outcome)
    update = k * grad
    new_ratings = ratings.at[comp_idxs[0]].add(update)
    new_ratings = new_ratings.at[comp_idxs[1]].add(-update)
    new_val = {
        'ratings': new_ratings,
        'matchups': prev_val['matchups'],
        'outcomes': prev_val['outcomes']
    }
    return new_val

def run_riix_elo(dataset, mode):
    elo = Elo(
        competitors=dataset.competitors,
        initial_rating=0.0,
        alpha=math.log(10.0) / 400.0,
        k=32.0,
        update_method=mode,
    )
    elo.fit_dataset(dataset)
    return elo.ratings


@jax.jit
def do_update(stale_ratings, fresh_ratings):
    return jnp.copy(fresh_ratings), jnp.copy(fresh_ratings)

@jax.jit
def do_nothing(stale_ratings, fresh_ratings):
    return stale_ratings, fresh_ratings

@partial(jax.jit, static_argnums=(2,))
def batched_elo_update(carry, x, alpha, k):
    stale_ratings = carry['stale_ratings']
    fresh_ratings = carry['fresh_ratings']
    prev_time_step = carry['prev_time_step']
    comp_idxs = x['matchups']
    cur_time_step = x['time_steps']
    outcome = x['outcomes']

    stale_ratings, fresh_ratings = jax.lax.cond(
        cur_time_step != prev_time_step,
        do_update,
        do_nothing,
        stale_ratings,
        fresh_ratings,
    )
    comp_ratings = stale_ratings[comp_idxs]

    # grad = elo_grad(comp_ratings, outcome)
    grad = elo_grad_hard(comp_ratings, outcome, alpha)
    update = k * grad
    fresh_ratings = fresh_ratings.at[comp_idxs[0]].add(update)
    fresh_ratings = fresh_ratings.at[comp_idxs[1]].add(-update)

    new_carry = {
        'stale_ratings': stale_ratings,
        'fresh_ratings': fresh_ratings,
        'prev_time_step': cur_time_step
    }
    return new_carry, None

def run_online_raax_elo(matchups, outcomes, num_competitors, alpha=math.log(10.0)/400.0, k=32.0):
    ratings = jnp.zeros(num_competitors, dtype=jnp.float64)
    init_val = {
        'matchups': matchups,
        'outcomes': outcomes,
        'ratings': ratings,
    }
    lower = 0
    upper = matchups.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=partial(online_elo_update, alpha=alpha, k=k),
        init_val=init_val,
    )
    new_ratings = final_val['ratings']
    return new_ratings

def run_batched_raax_elo(matchups, outcomes, time_steps, num_competitors, alpha=math.log(10.0) / 400.0, k=32.0):
    stale_ratings = jnp.zeros(num_competitors, dtype=jnp.float64)
    fresh_ratings = jnp.zeros(num_competitors, dtype=jnp.float64)
    xs = {
        'matchups': matchups,
        'outcomes': outcomes,
        'time_steps': time_steps,
    }
    init_val = {
        'stale_ratings': stale_ratings,
        'fresh_ratings': fresh_ratings,
        'prev_time_step': 0,
    }
    final_val, _ = jax.lax.scan(
        f=partial(batched_elo_update, alpha=alpha, k=k),
        init=init_val,
        xs=xs,
        length=matchups.shape[0],
    )
    return final_val['fresh_ratings']



if __name__ == '__main__':
    # dataset = get_dataset("league_of_legends", '7D')
    # dataset = get_dataset("starcraft2", '1D')
    dataset = get_dataset("smash_melee", '1D')

    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)

    online_riix_ratings = time_function(
        partial(run_riix_elo, dataset, 'iterative'), 'online riix elo', 10
    )
    online_raax_ratings = time_function(
        partial(run_online_raax_elo, matchups, outcomes, dataset.num_competitors), 'online raax elo', 10
    )

    print('----------------------------')
    batched_riix_ratings = time_function(
        partial(run_riix_elo, dataset, 'batched'), 'batched riix elo', 10
    )
    batched_raax_ratings = time_function(
        partial(run_batched_raax_elo, matchups, outcomes, time_steps, dataset.num_competitors), 'batched raax elo', 10
    )

    print('online diffs:')
    print(np.min(np.abs(online_riix_ratings - online_raax_ratings)))
    print(np.max(np.abs(online_riix_ratings - online_raax_ratings)))
    print(np.mean(np.abs(online_riix_ratings - online_raax_ratings)))
    print('batched diffs:')
    print(np.min(np.abs(batched_riix_ratings - batched_raax_ratings)))
    print(np.max(np.abs(batched_riix_ratings - batched_raax_ratings)))
    print(np.mean(np.abs(batched_riix_ratings - batched_raax_ratings)))
