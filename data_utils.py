import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from datasets import load_dataset

def gimmie_data(game):
    df = load_dataset('EsportsBench/EsportsBench', split=game).to_pandas()
    dataset = MatchupDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date'
    )
    return jnp.asarray(dataset.matchups), jnp.asarray(dataset.outcomes)
