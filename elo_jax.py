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

def elo_update(idx, prev_val):
    ratings = prev_val['ratings']
    running_grads = prev_val['running_grads']
    comp_idxs = prev_val['schedule'][idx,1:]
    comp_ratings = ratings[comp_idxs]
    outcome = prev_val['outcomes'][idx]
    grad = elo_grad(comp_ratings, outcome)
    new_ratings = ratings.at[comp_idxs].add(grad)
    new_val = {
        'ratings': new_ratings,
        'running_grads': running_grads,
        'schedule': prev_val['schedule'],
        'outcomes': prev_val['outcomes']
    }
    return new_val


def run_riix_elo(dataset):
    elo = Elo(competitors=dataset.competitors)
    elo.fit_dataset(dataset)
    return elo.ratings

def run_raax_elo(dataset):
    C = dataset.num_competitors
    N = len(dataset)
    time_steps = jnp.array(dataset.time_steps)
    matchups = jnp.array(dataset.matchups)
    schedule = jnp.column_stack((time_steps, matchups))
    outcomes = jnp.array(dataset.outcomes)
    ratings = jnp.zeros(C, dtype=jnp.float32)
    running_grads = jnp.zeros(C)
    init_val = {
        'schedule': schedule,
        'outcomes': outcomes,
        'ratings': ratings,
        'running_grads' : running_grads,
    }
    lower = 0
    upper = schedule.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=elo_update,
        init_val=init_val,
    )
    new_ratings = np.asarray(final_val['ratings'])
    return new_ratings


if __name__ == '__main__':
    # C = 4
    # N = 10

    # schedule = jnp.array(
    #     [[0, 0, 1],
    #      [0, 1, 2],
    #      [1, 1, 2],
    #      [2, 0, 3],
    #      [2, 3, 1],
    #      [2, 3, 0],
    #      [3, 2, 0],
    #      [5, 2, 3],
    #      [5, 3, 1],
    #      [5, 1, 0]]
    # )
    # outcomes = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    start_time = time.time()
    # split = 'league_of_legends'
    split = 'starcraft2'
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
    riix_ratings = run_riix_elo(dataset)
    print(f'riix fit time: {time.time() - start_time}')

    start_time = time.time()
    raxx_ratings = run_raax_elo(dataset)
    print(f'raax fit time: {time.time() - start_time}')

    print(riix_ratings)
    print(raxx_ratings)
