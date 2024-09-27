
import jax
import jax.numpy as jnp

@jax.jit
def elo_grad_hard(ratings, outcome):
    prob = jax.nn.sigmoid(ratings[0] - ratings[1])
    grad = outcome - prob
    return jnp.array([grad, -grad])

@jax.jit
def batched_elo_update(carry, x):
    ratings = carry['ratings']
    running_grads = carry['running_grads']
    update_mask = x['update_mask']

    # Apply updates using vectorized operations
    ratings = ratings + update_mask * running_grads
    running_grads = running_grads * (1 - update_mask)

    comp_idxs = x['schedule'][1:]
    comp_ratings = ratings[comp_idxs]
    outcome = x['outcomes']
    grad = elo_grad_hard(comp_ratings, outcome)
    
    # Create a gradient array of the same shape as ratings
    full_grad = jnp.zeros_like(ratings)
    full_grad = full_grad.at[comp_idxs].set(grad)
    
    new_running_grads = running_grads + full_grad

    new_carry = {
        'ratings': ratings,
        'running_grads': new_running_grads,
    }
    return new_carry, None

@jax.jit
def run_batched_elo(initial_ratings, schedules, outcomes, update_masks):
    def scan_fn(carry, x):
        return batched_elo_update(carry, {
            'schedule': x[0],
            'outcomes': x[1],
            'update_mask': x[2]
        })

    initial_carry = {
        'ratings': initial_ratings,
        'running_grads': jnp.zeros_like(initial_ratings)
    }

    final_carry, _ = jax.lax.scan(
        scan_fn,
        initial_carry,
        (schedules, outcomes, update_masks)
    )

    return final_carry['ratings']

# Example usage
initial_ratings = jnp.array([1500.0, 1600.0, 1400.0, 1550.0])
schedules = jnp.array([[0, 1], [2, 3], [1, 2], [0, 3]])
outcomes = jnp.array([1.0, 0.0, 0.5, 1.0])
update_masks = jnp.array([0, 0, 0, 1])  # Update on the last iteration

final_ratings = run_batched_elo(initial_ratings, schedules, outcomes, update_masks)
print("Final ratings:", final_ratings)