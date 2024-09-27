import os
import jax.numpy as jnp
import polars as pl
from riix.utils.data_utils import MatchupDataset
from datasets import load_dataset

def load_dataset(game, rating_period='7D'):
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

