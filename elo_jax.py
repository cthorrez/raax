import jax
import jax.numpy as jnp
from datasets import load_dataset
from riix.utils.data_utils import MatchupDataset

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
        'schedule': schedule,
        'outcomes': outcomes
    }
    return new_val

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


    lol = load_dataset("EsportsBench/EsportsBench", split="league_of_legends").to_pandas()
    dataset = MatchupDataset(
        df=lol,
        competitor_cols=["competitor_1", "competitor_2"],
        outcome_col="outcome",
        datetime_col="date",
        rating_period="7D"
    )

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
    new_ratings = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=elo_update,
        init_val=init_val,
    )
    print(new_ratings)