import jax
import jax.numpy as jnp

class RatingSystem:
    def __init__(self, competitors, update_fn):
        self.competitors = competitors
        self.num_competitors = len(competitors)
        self.ratings = jnp.zeros(shape=self.num_competitors, dtype=jnp.float32)
        self.update_fn = update_fn

    def fit(self, time_steps, schedule, outcomes):
        final_state = jax.lax.scan(
            f = self.update_fn,
            init = self.ratings,
            xs = {
                't' : None,
                'schedule' : schedule,
                'outcomes' : outcomes,
            }
        )

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def elo_loss(idx, c, ratings, schedule, outcomes):
    
    matchups = jax.lax.dynamic_slice_in_dim(
        schedule[:,1:],
        start_index=idx,
        slice_size=c,
        axis=0,
    )
    t_outcomes = outcomes[idx:idx+c]

    matchup_ratings = ratings[matchups]
    logits = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(logits)
    loss = t_outcomes * jnp.log(probs) + (1.0 - t_outcomes) * jnp.log(1.0 - probs)
    return loss

elo_grad = jax.grad(elo_loss, argnums=(2,))

def elo_update(carry, idx):
    ratings, schedule, outcomes, ts, t_idxs, counts = carry
    t_idx = t_idxs[idx]
    c = counts[idx]
    grad = elo_grad(t_idx, c, ratings, schedule=schedule, outcomes=outcomes)
    new_ratings = ratings + grad
    return (new_ratings, schedule, outcomes), None

if __name__ == '__main__':
    C = 4
    N = 10

    ratings = jnp.zeros(C, dtype=jnp.float32)
    schedule = jnp.array(
        [[0, 0, 1],
         [0, 1, 2],
         [1, 1, 2],
         [2, 0, 3],
         [2, 3, 1],
         [2, 3, 0],
         [3, 2, 0],
         [5, 2, 3],
         [5, 3, 1],
         [5, 1, 0]]
    )
    outcomes = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    ts, t_idxs, _, counts  = jnp.unique_all(schedule[:,0])
    

    
    init = (ratings, schedule, outcomes, ts, t_idxs, counts)
    xs = jnp.arange(ts.shape[0])

    _, new_ratings = jax.lax.scan(elo_update, init, xs)
    print(new_ratings)



