import math
import time
from functools import partial
import numpy as np
from contextlib import contextmanager
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from data_utils import load_dataset
from riix.utils.data_utils import MatchupDataset
from riix.models.glicko import Glicko

# I think jax is smart enough for this
@partial(jax.jit, static_argnums=(1,))
def g(rd, three_q2_over_pi2):
    return 1.0 / jnp.sqrt(1.0 + ((rd**2.0) * three_q2_over_pi2))

def run_online_glicko(
    matchups,
    outcomes,
    update_mask,
    num_competitors,
    initial_mu=1500.0,
    initial_rd=350.0,
    c=63.2,
    q=math.log(10.0)/400.0,
):
    c2 = c ** 2.0
    three_q2_over_pi2 = (3.0 * q**2.0) / (math.pi ** 2.0)
    mus = jnp.full(shape=(num_competitors,), fill_value=initial_mu, dtype=jnp.float64)
    rds = jnp.full(shape=(num_competitors,), fill_value=initial_rd, dtype=jnp.float64)
    error = jnp.zeros(shape=(num_competitors,), dtype=jnp.float64)
    d2_inv = jnp.zeros(shape=(num_competitors,), dtype=jnp.float64)

    init_val = {
        'matchups': matchups,
        'outcomes': outcomes,
        'mus': mus,
        'rds': rds,
        'error': error,
        'd2_inv': d2_inv,

    }
    lower = 0
    upper = matchups.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=partial(online_elo_update, alpha=alpha, k=k),
        init_val=init_val,
    )
    new_ratings = final_val['ratings']
    return new_ratings