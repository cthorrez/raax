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
    
    def update(self, c_idxs, time_step, outcome, state, **kwargs):
        raise NotImplementedError
    
    def _update(self, state, x, **kwargs):
        c_idxs, time_step, outcome = x
        new_state, prob = self.update(c_idxs, time_step, outcome, state, **kwargs)
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
    
    def sweep2(self, matches, time_steps, outcomes, sweep_params):
        # First, create a vectorized version of the update function that operates
        # across all parameter combinations at once
        @partial(jax.vmap, in_axes=(0, 0, None, None, None, None))
        def batch_update(states, param_dicts, idx_a, idx_b, time_step, outcome):
            new_state, prob = self.update(idx_a, idx_b, time_step, outcome, states, **param_dicts)
            return new_state, prob

        # Prepare initial states for all parameter combinations
        param_keys = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        num_combinations = len(param_values[0])
        
        # Create a list of parameter dictionaries for each combination,
        # starting with the base params and updating with sweep params
        param_dicts = []
        for i in range(num_combinations):
            params = self.params.copy()  # Start with base params
            params.update({k: sweep_params[k][i] for k in param_keys})  # Update with sweep params
            param_dicts.append(params)
        
        # Convert param_dicts into a dictionary of arrays
        param_arrays = {k: jnp.array([d[k] for d in param_dicts]) for k in self.params.keys()}
        
        # Initialize states for all parameter combinations using vmapped get_init_state
        init_states = jax.vmap(self.get_init_state)(**param_arrays)
        
        # Define the scan function that will apply batch_update at each time step
        def scan_fn(carry, x):
            states, param_arrays = carry
            (idx_a, idx_b), time_step, outcome = x
            new_states, probs = batch_update(states, param_arrays, idx_a, idx_b, time_step, outcome)
            return (new_states, param_arrays), probs

        # Run the scan over time steps with batched parameters
        (final_states, _), many_probs = jax.lax.scan(
            f=scan_fn,
            init=(init_states, param_arrays),
            xs=(matches, time_steps, outcomes)
        )

        # Transpose many_probs once to get shape (n_params, n_matches)
        many_probs = many_probs.T
        
        # Compute metrics using the transposed probs
        loss = log_loss(many_probs, jnp.expand_dims(outcomes, 0), axis=1)
        acc = accuracy(many_probs, jnp.expand_dims(outcomes, 0), axis=1)
        best_idx = jnp.nanargmax(acc)

        # Print results (keeping the same format as original sweep)
        print('best metrics:')
        print(f'acc: {acc[best_idx].item():.4f}')
        print(f'log loss: {loss[best_idx].item():.4f}')
        print('best params:')
        for param, vals in sweep_params.items():
            print(f'  {param}: {vals[best_idx]}')

        return final_states, many_probs, best_idx

if __name__ == '__main__':
    pass
