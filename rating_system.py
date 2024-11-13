import jax
import jax.numpy as jnp

class OnlineRatingSystem:
    def __init__(self, competitors):
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def initialize_state(self):
        raise NotImplementedError()
    
    def update(self, idx_a, idx_b, time_step, outcome, state):
        raise NotImplementedError
    
    def _update(self, state, x):
        (idx_a, idx_b), time_step, outcome = x
        new_state, prob = self.update(idx_a, idx_b, time_step, outcome, state)
        return new_state, prob
        

    def fit(self, matches, time_steps, outcomes):
        init_state = self.initialize_state()
        final_state, probs = jax.lax.scan(
            f = self._update,
            init = init_state,
            xs = (
                matches,
                time_steps,
                outcomes
            )
        )
        return final_state, probs



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



