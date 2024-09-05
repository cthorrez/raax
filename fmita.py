import jax
import jax.numpy as jnp

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64
# dtype = jnp.float32

key = jax.random.PRNGKey(0)
logits = jax.random.uniform(key, shape=(100,), minval=-10.0, maxval=10.0, dtype=dtype)

# Generate random outcomes (0 or 1) from a discrete uniform distribution
outcomes = jax.random.randint(key, shape=(100,), minval=0, maxval=2, dtype=jnp.int32)

# Define the four loss expressions with fp64 precision
def loss1(logit, outcome):
    prob = jax.nn.sigmoid(logit)
    return outcome * jnp.log(prob) + (1.0 - outcome) * jnp.log(1.0 - prob)

def loss2(logit, outcome):
    return jax.nn.softplus(jnp.where(outcome, logit, -logit))

def loss3(logit, outcome):
    return -jax.nn.softplus(jnp.where(outcome, -logit, logit))

def loss4(logit, outcome):
    return jax.nn.softplus(jnp.where(outcome, -logit, logit))

# Compute the losses
l1 = loss1(logits, outcomes)
l2 = loss2(logits, outcomes)
l3 = loss3(logits, outcomes)
l4 = loss4(logits, outcomes)

# Calculate and print the mean absolute difference between each pair of loss functions
def mean_abs_diff(loss_a, loss_b):
    return jnp.max(jnp.abs(loss_a - loss_b))

print(f"Mean absolute diff between loss1 and loss2: {mean_abs_diff(l1, l2)}")
print(f"Mean absolute diff between loss1 and loss3: {mean_abs_diff(l1, l3)}")
print(f"Mean absolute diff between loss1 and loss4: {mean_abs_diff(l1, l4)}")
print(f"Mean absolute diff between loss2 and loss3: {mean_abs_diff(l2, l3)}")
print(f"Mean absolute diff between loss2 and loss4: {mean_abs_diff(l2, l4)}")
print(f"Mean absolute diff between loss3 and loss4: {mean_abs_diff(l3, l4)}")