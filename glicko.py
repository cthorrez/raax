import math
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from utils import timer
from data_utils import load_dataset, jax_preprocess

from riix.models.glicko import Glicko

@partial(jax.jit, static_argnums=(1,))
def g(rd_squared, three_q2_over_pi2):
    return 1.0 / jnp.sqrt(1.0 + ((rd_squared) * three_q2_over_pi2))

SIGN_FLIP = jnp.array([1.0, -1.0])

@partial(jax.jit, static_argnums=(2,3,4))
def online_glicko_update(idx, prev_val, max_rd, c2, three_q2_over_pi2):
    mus = prev_val['mus']
    rds = prev_val['rds']
    error = prev_val['error']
    d2_inv = prev_val['d2_inv']
    comp_idxs = prev_val['matchups'][idx]
    outcome = prev_val['outcomes'][idx]

    cur_mus = mus[comp_idxs]
    cur_rds = rds[comp_idxs]

    cur_rds_squared = jnp.square(cur_rds)
    cur_gs = g(cur_rds_squared, three_q2_over_pi2)

    mu_diff = (cur_mus * SIGN_FLIP).sum() * SIGN_FLIP

    new_val = {
        'matchups': prev_val['matchups'],
        'outcomes': prev_val['outcomes'],
        'mus': mus,
        'rds': rds,
        'error': error,
        'd2_inv': d2_inv,
    }
    return new_val

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
        body_fun=partial(
            online_glicko_update,
            max_rd=initial_rd,
            c2=c2,
            three_q2_over_pi2=three_q2_over_pi2
        ),
        init_val=init_val,
    )
    new_ratings = final_val['mus']
    return new_ratings

def main():
    dataset = load_dataset("smash_melee", '1D')
    matchups, outcomes, update_mask, start_idxs, end_idxs = jax_preprocess(dataset)

    with timer('rax online glicko'):
        mus = run_online_glicko(matchups, outcomes, update_mask, num_competitors=dataset.num_competitors)
    print(mus)

if __name__ == '__main__':
    main()