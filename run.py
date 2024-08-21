import jax
import jax.numpy as jnp
from riix.utils.data_utils import MatchupDataset
from datasets import load_dataset


def main():
    lol = load_dataset("EsportsBench/EsportsBench", split="league_of_legends").to_pandas()
    dataset = MatchupDataset(
        df=lol,
        competitor_cols=["competitor_1", "competitor_2"],
        outcome_col="outcome",
        datetime_col="date",
        rating_period="7D"
    )
    batch_sizes = []
    for matchups, outcomes, t in dataset:
        batch_sizes.append(matchups.shape[0])
    print(jnp.min(jnp.array(batch_sizes)))
    print(jnp.max(jnp.array(batch_sizes)))
    print(jnp.mean(jnp.array(batch_sizes)))
    

if __name__ == '__main__':
    main()