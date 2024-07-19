import jax
import jax.numpy as jnp

class RatingSystem:
    def __init__(self, competitors, update_fn):
        self.competitors = competitors
        self.update_fn = update_fn

    def fit(self, time_steps, schedule, outcomes):
        pass
