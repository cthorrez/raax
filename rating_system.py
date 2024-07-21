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
                't' : None
                'schedule' : schedule,
                'outcomes' : outcomes,
            }
        )
