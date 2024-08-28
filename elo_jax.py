import time
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
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

def online_elo_update(idx, prev_val):
    ratings = prev_val['ratings']
    comp_idxs = prev_val['schedule'][idx,1:]
    comp_ratings = ratings[comp_idxs]
    outcome = prev_val['outcomes'][idx]
    grad = elo_grad(comp_ratings, outcome)
    new_ratings = ratings.at[comp_idxs].add(grad)
    new_val = {
        'ratings': new_ratings,
        'schedule': prev_val['schedule'],
        'outcomes': prev_val['outcomes']
    }
    return new_val

def online_elo_update(idx, prev_val):
    ratings = prev_val['ratings']
    comp_idxs = prev_val['schedule'][idx,1:]
    comp_ratings = ratings[comp_idxs]
    outcome = prev_val['outcomes'][idx]
    grad = elo_grad(comp_ratings, outcome)
    new_ratings = ratings.at[comp_idxs].add(grad)
    new_val = {
        'ratings': new_ratings,
        'schedule': prev_val['schedule'],
        'outcomes': prev_val['outcomes']
    }
    return new_val

def do_update(ratings, running_grads):
    return ratings + running_grads, jnp.zeros_like(running_grads)

def do_nothing(ratings, running_grads):
    return ratings, running_grads

def batched_elo_update(carry, x):
    ratings = carry['ratings']
    running_grads = carry['running_grads']
    update_flag = x['update_mask']

    ratings, running_grads = jax.lax.cond(
        update_flag,
        do_update,
        do_nothing,
        ratings,
        running_grads
    )
    # ratings = ratings + (update_flag * running_grads)
    # running_grads = running_grads * (1 - update_flag)
    comp_idxs = x['schedule'][1:]
    comp_ratings = ratings[comp_idxs]
    outcome = x['outcomes']
    grad = elo_grad(comp_ratings, outcome)
    new_running_grads = running_grads.at[comp_idxs].add(grad)


    new_carry = {
        'ratings': ratings,
        'running_grads': new_running_grads,
    }
    return new_carry, None


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

def run_online_raax_elo(dataset):
    C = dataset.num_competitors
    N = len(dataset)
    time_steps = jnp.array(dataset.time_steps)
    matchups = jnp.array(dataset.matchups)
    schedule = jnp.column_stack((time_steps, matchups))
    outcomes = jnp.array(dataset.outcomes)
    ratings = jnp.zeros(C, dtype=jnp.float32)
    init_val = {
        'schedule': schedule,
        'outcomes': outcomes,
        'ratings': ratings,
    }
    lower = 0
    upper = schedule.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=online_elo_update,
        init_val=init_val,
    )
    new_ratings = np.asarray(final_val['ratings'])
    return new_ratings

def run_batched_raax_elo(dataset):
    C = dataset.num_competitors
    time_steps = jnp.array(dataset.time_steps)
    update_mask = jnp.insert(jnp.diff(time_steps) != 0, 0, False)
    matchups = jnp.array(dataset.matchups)
    schedule = jnp.column_stack((time_steps, matchups))
    outcomes = jnp.array(dataset.outcomes)
    ratings = jnp.zeros(C, dtype=jnp.float32)
    running_grads = jnp.zeros(C)
    xs = {
        'schedule': schedule,
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
    new_ratings = np.asarray(final_val['ratings']) + np.asarray(final_val['running_grads'])
    return new_ratings


if __name__ == '__main__':
    start_time = time.time()
    split = 'league_of_legends'
    # split = 'starcraft2'
    split='tetris'
    lol = load_dataset("EsportsBench/EsportsBench", split=split).to_pandas()
    dataset = MatchupDataset(
        df=lol,
        competitor_cols=["competitor_1", "competitor_2"],
        outcome_col="outcome",
        datetime_col="date",
        rating_period="7D"
    )
    print(f'data load duration: {time.time() - start_time}')

    start_time = time.time()
    online_riix_ratings = run_riix_elo(dataset, 'iterative')
    print(f'online riix fit time: {time.time() - start_time}')

    start_time = time.time()
    online_raxx_ratings = run_online_raax_elo(dataset)
    print(f'online raax fit time: {time.time() - start_time}')

    start_time = time.time()
    batched_riix_ratings = run_riix_elo(dataset, 'batched')
    print(f'batched riix fit time: {time.time() - start_time}')

    start_time = time.time()
    batched_raxx_ratings = run_batched_raax_elo(dataset)
    print(f'batched raax fit time: {time.time() - start_time}')

    print('online diffs:')
    print(np.min(np.abs(online_riix_ratings - online_raxx_ratings)))
    print(np.max(np.abs(online_riix_ratings - online_raxx_ratings)))
    print(np.mean(np.abs(online_riix_ratings - online_raxx_ratings)))

    print('batched diffs:')
    print(np.min(np.abs(batched_riix_ratings - batched_raxx_ratings)))
    print(np.max(np.abs(batched_riix_ratings - batched_raxx_ratings)))
    print(np.mean(np.abs(batched_riix_ratings - batched_raxx_ratings)))

