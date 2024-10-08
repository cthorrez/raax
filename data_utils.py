import os
import jax
import jax.numpy as jnp
import polars as pl
from riix.utils.data_utils import MatchupDataset
from datasets import load_dataset

def get_dataset(game, rating_period='7D'):
    if os.path.exists(f'data/{game}.parquet'):
        df = pl.read_parquet(f'data/{game}.parquet').to_pandas()
    else:
        os.makedirs('data', exist_ok=True)
        df = load_dataset('EsportsBench/EsportsBench', split=game).to_polars()
        df.write_parquet(f'data/{game}.parquet')
        df = df.to_pandas()
    dataset = MatchupDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period=rating_period,
    )
    return dataset

def jax_preprocess(dataset):
    time_steps = jnp.array(dataset.time_steps)
    update_mask = jnp.insert(jnp.diff(time_steps) != 0, 0, False)
    matchups = jnp.array(dataset.matchups)
    outcomes = jnp.array(dataset.outcomes)
    max_competitors_per_timestep = get_max_competitors_per_timestep(matchups, time_steps)
    return matchups, outcomes, time_steps, update_mask, max_competitors_per_timestep


@jax.jit
def do_update_fn(time_step_counts, per_time_step_mask, time_step):
    time_step_counts = time_step_counts.at[time_step].set(jnp.sum(per_time_step_mask).astype(jnp.int32))
    return time_step_counts, jnp.zeros_like(per_time_step_mask)

@jax.jit
def do_nothing_fn(time_step_counts, per_time_step_mask, time_step):
    return time_step_counts, per_time_step_mask

@jax.jit
def body_fn(idx, prev_val):
    matchup = prev_val['matchups'][idx]
    time_step = prev_val['time_steps'][idx]
    prev_time_step = prev_val['prev_time_step']
    time_step_counts = prev_val['time_step_counts']
    per_time_step_mask = prev_val['per_time_step_mask']

    update_flag = time_step != prev_time_step
    time_step_counts, per_time_step_mask = jax.lax.cond(
        update_flag,
        do_update_fn,
        do_nothing_fn,
        time_step_counts,
        per_time_step_mask,
        time_step,
    )
    per_time_step_mask = per_time_step_mask.at[matchup[0],].set(1)
    per_time_step_mask = per_time_step_mask.at[matchup[1],].set(1)


    return {
        'matchups': prev_val['matchups'],
        'time_steps': prev_val['time_steps'],
        'time_step_counts': time_step_counts,
        'per_time_step_mask': per_time_step_mask,
        'prev_time_step': time_step,
    }



def get_max_competitors_per_timestep(matchups, time_steps):
    num_competitors = jnp.max(matchups) + 1
    time_step_counts = jnp.zeros(shape=(time_steps.shape[0],), dtype=jnp.int32)
    per_time_step_mask = jnp.zeros(shape=(num_competitors,), dtype=jnp.int32)
    init_val = {
        'matchups': matchups,
        'time_steps': time_steps,
        'time_step_counts': time_step_counts,
        'per_time_step_mask': per_time_step_mask,
        'prev_time_step': 0,
    }
    final_val = jax.lax.fori_loop(
        lower=0,
        upper=time_steps.shape[0],
        body_fun=body_fn,
        init_val=init_val,
    )
    max_competitors_per_timestep = jnp.max(final_val['time_step_counts'])
    return max_competitors_per_timestep


