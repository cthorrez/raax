import time
from functools import partial
import numpy as np
from contextlib import contextmanager
import jax
import jax.numpy as jnp
from data_utils import load_dataset
from riix.utils.data_utils import MatchupDataset
from riix.models.elo import Elo

jax.default_device = jax.devices("cpu")[0]

@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name} duration (s): {end_time - start_time:.4f}")

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def elo_loss(ratings, outcome):
    logit = (ratings * jnp.array([1.0, -1.0])).sum()
    prob = sigmoid(logit)
    loss = outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)
    return loss

elo_grad = jax.grad(elo_loss, argnums=0)

@jax.jit
def elo_grad_hard(ratings, outcome):
    prob = jax.nn.sigmoid(ratings[0] - ratings[1])
    grad = outcome - prob
    return grad

vec_elo_grad_hard = jax.jit(jax.vmap(elo_grad_hard))


def online_elo_update(idx, prev_val):
    ratings = prev_val['ratings']
    comp_idxs = prev_val['matchups'][idx]
    comp_ratings = ratings[comp_idxs]
    outcome = prev_val['outcomes'][idx]
    grad = elo_grad_hard(comp_ratings, outcome)
    # grad = elo_grad(comp_ratings, outcome)
    new_ratings = ratings.at[comp_idxs[0]].add(grad)
    new_ratings = new_ratings.at[comp_idxs[1]].add(-grad)
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
        alpha=1.0,
        k=1.0,
        update_method=mode,
    )
    elo.fit_dataset(dataset)
    return elo.ratings

@jax.jit
def do_update(ratings, running_grads):
    return ratings + running_grads, running_grads.at[:].set(0)

@jax.jit
def do_nothing(ratings, running_grads):
    return ratings, running_grads

def batched_elo_update(carry, x):
    ratings = carry['ratings']
    running_grads = carry['running_grads']
    comp_idxs = x['matchups']
    update_flag = x['update_mask']
    outcome = x['outcomes']

    ratings, running_grads = jax.lax.cond(
        update_flag,
        do_update,
        do_nothing,
        ratings,
        running_grads
    )

    # ratings = ratings + (update_flag * running_grads)
    # running_grads = running_grads * (1 - update_flag)

    comp_ratings = ratings[comp_idxs]

    # grad = elo_grad(comp_ratings, outcome)
    grad = elo_grad_hard(comp_ratings, outcome)
    new_running_grads = running_grads.at[comp_idxs[0]].add(grad)
    new_running_grads = new_running_grads.at[comp_idxs[1]].add(-grad)

    new_carry = {
        'ratings': ratings,
        'running_grads': new_running_grads,
    }
    return new_carry, None

def run_online_raax_elo(matchups, outcomes, num_competitors):
    ratings = jnp.zeros(num_competitors, dtype=jnp.float32)
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
        body_fun=online_elo_update,
        init_val=init_val,
    )
    new_ratings = final_val['ratings']
    return new_ratings

def run_batched_raax_elo(matchups, outcomes, update_mask, num_competitors):
    ratings = jnp.zeros(num_competitors, dtype=jnp.float32)
    running_grads = jnp.zeros(num_competitors, dtype=jnp.float32)
    xs = {
        'matchups': matchups,
        'outcomes': outcomes,
        'update_mask': update_mask,
    }
    init_val = {
        'ratings': ratings,
        'running_grads': running_grads,
    }
    final_val, _ = jax.lax.scan(
        f=batched_elo_update,
        init=init_val,
        xs=xs,
    )
    new_ratings = final_val['ratings'] + final_val['running_grads']
    return new_ratings

# def run_vmap_raax_elo(matchups, outcomes, start_idxs, end_idxs, num_competitors):
#     ratings = jnp.zeros(num_competitors, dtype=jnp.float32)
#     for start_idx, end_idx in zip(start_idxs, end_idxs):
#         print(start_idx, end_idx)
#         period_matchups = matchups[start_idx:end_idx,:]
#         period_outcomes = outcomes[start_idx:end_idx]
#         period_ratings = ratings[period_matchups]
#         probs = jax.nn.sigmoid(period_ratings[:,0] - period_ratings[:,1])
#         updates = period_outcomes - probs
#         ratings = ratings.at[period_matchups[:,0]].add(updates)
#         ratings = ratings.at[period_matchups[:,1]].add(-updates)
#     return ratings

with jax.disable_jit():
    def update_ratings(ratings, matchups, outcomes):
        player1_ratings = ratings[matchups[:,0]]
        player2_ratings = ratings[matchups[:,1]]
        rating_diff = player1_ratings - player2_ratings
        probs = jax.nn.sigmoid(rating_diff)
        updates = outcomes - probs
        
        # Prepare indices and updates for scatter_add
        indices = jnp.concatenate([matchups[:,0], matchups[:,1]])
        all_updates = jnp.concatenate([updates, -updates])
        
        # Use scatter_add to apply all updates at once
        return ratings.at[indices].add(all_updates)


    def run_vmap_raax_elo(matchups, outcomes, start_idxs, end_idxs, num_competitors):
        ratings = jnp.zeros(num_competitors, dtype=jnp.float32)
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            period_matchups = matchups[start_idx:end_idx,:]
            period_outcomes = outcomes[start_idx:end_idx]
            ratings = update_ratings(ratings, period_matchups, period_outcomes)
        return ratings

def jax_preprocess(dataset):
    time_steps = jnp.array(dataset.time_steps)
    unique_times, counts = jnp.unique_counts(time_steps)
    end_idxs = jnp.cumsum(counts)
    start_idxs = jnp.concatenate([jnp.array([0]), end_idxs[:-1]])

    update_mask = jnp.insert(jnp.diff(time_steps) != 0, 0, False)
    matchups = jnp.array(dataset.matchups)
    outcomes = jnp.array(dataset.outcomes)
    return matchups, outcomes, update_mask, start_idxs, end_idxs


if __name__ == '__main__':
    # dataset = load_dataset("league_of_legends", '1D')
    dataset = load_dataset("tetris", '1D')

    matchups, outcomes, update_mask, start_idxs, end_idxs = jax_preprocess(dataset)

    with timer('iterative riix'):
        online_riix_ratings = run_riix_elo(dataset, 'iterative')
    with timer('online raax'):
        online_raax_ratings = run_online_raax_elo(matchups, outcomes, dataset.num_competitors)
    with timer('batched riix'):
        batched_riix_ratings = run_riix_elo(dataset, 'batched')
    with timer('batched raax'):
        batched_raax_ratings = run_batched_raax_elo(matchups, outcomes, update_mask, dataset.num_competitors)
    with timer('vmap raax'):
        with jax.disable_jit():
            vmap_raax_ratings = run_vmap_raax_elo(matchups, outcomes, start_idxs, end_idxs, dataset.num_competitors)

    print('online diffs:')
    print(np.min(np.abs(online_riix_ratings - online_raax_ratings)))
    print(np.max(np.abs(online_riix_ratings - online_raax_ratings)))
    print(np.mean(np.abs(online_riix_ratings - online_raax_ratings)))
    print('batched diffs:')
    print(np.min(np.abs(batched_riix_ratings - batched_raax_ratings)))
    print(np.max(np.abs(batched_riix_ratings - batched_raax_ratings)))
    print(np.mean(np.abs(batched_riix_ratings - batched_raax_ratings)))
    print('vmap diffs:')
    print(np.min(np.abs(batched_riix_ratings - vmap_raax_ratings)))
    print(np.max(np.abs(batched_riix_ratings - vmap_raax_ratings)))
    print(np.mean(np.abs(batched_riix_ratings - vmap_raax_ratings)))