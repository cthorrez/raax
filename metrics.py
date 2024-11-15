import jax.numpy as jnp

def log_loss(probs, outcomes, axis=0):
    return -(outcomes * jnp.log(probs) + (1.0 - outcomes) * jnp.log(1.0 - probs)).mean(axis=axis)

def accuracy(probs, outcomes, axis=0):
    corr = 0.0
    corr += ((probs > 0.5) & (outcomes == 1.0)).astype(jnp.float32).sum(axis=axis)
    corr += ((probs < 0.5) & (outcomes == 0.0)).astype(jnp.float32).sum(axis=axis)
    corr += (0.5 * (probs == 0.5).astype(jnp.float32)).sum(axis=axis)
    return corr / outcomes.shape[axis]
