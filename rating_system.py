import jax
import jax.numpy as jnp
from functools import partial
from metrics import log_loss, accuracy

class OnlineRatingSystem:
    def __init__(self, competitors):
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def get_init_state(self, **kwargs):
        raise NotImplementedError()
    
    def update(self, idx_a, idx_b, time_step, outcome, state, **kwargs):
        raise NotImplementedError
    
    def _update(self, state, x, **kwargs):
        (idx_a, idx_b), time_step, outcome = x
        new_state, prob = self.update(idx_a, idx_b, time_step, outcome, state, **kwargs)
        return new_state, prob

    def fit(self, matches, time_steps, outcomes, **overrides):
        merged_params = {**self.params, **overrides}
        init_state = self.get_init_state(**merged_params)
        final_state, probs = jax.lax.scan(
            f = partial(self._update, **merged_params),
            init = init_state,
            xs = (
                matches,
                time_steps,
                outcomes
            )
        )
        return final_state, probs
    
    
    def sweep(self, matches, time_steps, outcomes, sweep_params):
  
        @jax.jit
        def run_single(matches, time_steps, outcomes, params):
            final_state, probs = self.fit(matches, time_steps, outcomes, **params)
            return final_state, probs

        in_axes = (None, None, None, {param: 0 for param in sweep_params})
        run_many = jax.jit(jax.vmap(run_single, in_axes=in_axes))

        many_states, many_probs = run_many(matches, time_steps, outcomes, sweep_params)
        loss = log_loss(many_probs, jnp.expand_dims(outcomes, 0), axis=1)
        acc = accuracy(many_probs, jnp.expand_dims(outcomes, 0), axis=1)
        # best_idx = jnp.nanargmin(loss)
        best_idx = jnp.nanargmax(acc)
        print('best metrics:')
        print(f'acc: {acc[best_idx].item():.4f}')
        print(f'log loss: {loss[best_idx].item():.4f}')
        print('best params:')
        for param, vals in sweep_params.items():
            print(f'  {param}: {vals[best_idx]}')
        return many_states, many_probs, best_idx

if __name__ == '__main__':
    pass
