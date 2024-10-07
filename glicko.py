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

@partial(jax.jit, static_argnums=(2,3,4,5,6))
def online_glicko_update(idx, prev_val, max_rd, c2, q, q2, three_q2_over_pi2):
    mus = prev_val['mus']
    rds = prev_val['rds']

    comp_idxs = prev_val['matchups'][idx]
    outcome = prev_val['outcomes'][idx]

    cur_mus = mus[comp_idxs]
    cur_rds = rds[comp_idxs]

    cur_rds = jnp.minimum(max_rd, jnp.sqrt(jnp.square(cur_rds) + c2))
    cur_rds_squared = jnp.square(cur_rds)
    cur_gs = g(cur_rds_squared, three_q2_over_pi2)
    mu_diffs = (cur_mus * SIGN_FLIP).sum() * SIGN_FLIP
    probs = jax.nn.sigmoid(q * jnp.flip(cur_gs) * mu_diffs)
    d2_inv = q2 * jnp.square(jnp.flip(cur_gs)) * probs * (1.0 - probs)

    both_outcomes = jnp.array([outcome, 1.0 - outcome])
    mu_update_num = q * jnp.flip(cur_gs) * (both_outcomes - probs)
    mu_update_denom = (1.0 / jnp.square(cur_rds)) + d2_inv
    mu_updates = mu_update_num / mu_update_denom
    new_rds = jnp.sqrt(1.0 / mu_update_denom)
 
    mus = mus.at[comp_idxs].add(mu_updates)
    rds = rds.at[comp_idxs].set(new_rds)

    new_val = {
        'matchups': prev_val['matchups'],
        'outcomes': prev_val['outcomes'],
        'mus': mus,
        'rds': rds,
    }
    return new_val


@jax.jit
def do_update(mus, rds, sum_1, sum_2, q, q2):
    mu_update_num = q * sum_1
    d2_inv = q2 * sum_2
    mu_update_denom = (1.0 / jnp.square(rds)) + d2_inv
    mu_updates = mu_update_num / mu_update_denom
    mus = mus + mu_updates
    rds = jnp.sqrt(1.0 / mu_update_denom)
    return mus, rds, jnp.zeros_like(sum_1), jnp.zeros_like(sum_2)
    

@jax.jit
def do_nothing(mus, rds, sum_1, sum_2, q, q2):
    return mus, rds, sum_1, sum_2


@partial(jax.jit, static_argnums=(2,3,4,5,6))
def batched_glicko_update(idx, prev_val, max_rd, c2, q, q2, three_q2_over_pi2):
    mus = prev_val['mus']
    rds = prev_val['rds']
    sum_1 = prev_val['sum_1']
    sum_2 = prev_val['sum_2']
    comp_idxs = prev_val['matchups'][idx]
    outcome = prev_val['outcomes'][idx]
    update_flag = prev_val['update_mask'][idx]

    mus, rds, sum_1, sum_2 = jax.lax.cond(
        update_flag,
        do_update,
        do_nothing,
        mus,
        rds,
        sum_1,
        sum_2,
        q,
        q2,
    )

    cur_mus = mus[comp_idxs]
    cur_rds = rds[comp_idxs]

    cur_rds = jnp.minimum(max_rd, jnp.sqrt(jnp.square(cur_rds) + c2))

    cur_rds_squared = jnp.square(cur_rds)
    cur_gs = g(cur_rds_squared, three_q2_over_pi2)

    mu_diffs = (cur_mus * SIGN_FLIP).sum() * SIGN_FLIP
    probs = jax.nn.sigmoid(q * jnp.flip(cur_gs) * mu_diffs)
    
    both_outcomes = jnp.array([outcome, 1.0 - outcome])
    val_1 = jnp.flip(cur_gs) * (both_outcomes - probs)
    sum_1 = sum_1.at[comp_idxs].add(val_1)

    val_2 = jnp.square(jnp.flip(cur_gs)) * probs * (1.0 - probs)
    sum_2 = sum_2.at[comp_idxs].add(val_2)

    new_val = {
        'matchups': prev_val['matchups'],
        'outcomes': prev_val['outcomes'],
        'update_mask': prev_val['update_mask'],
        'mus': mus,
        'rds': rds,
        'sum_1': sum_1,
        'sum_2': sum_2,
    }
    return new_val

def run_online_glicko(
    matchups,
    outcomes,
    num_competitors,
    initial_mu=1500.0,
    initial_rd=350.0,
    c=0.0,
    # c=63.2,
    q=math.log(10.0)/400.0,
):
    c2 = c ** 2.0
    three_q2_over_pi2 = (3.0 * q**2.0) / (math.pi ** 2.0)
    mus = jnp.full(shape=(num_competitors,), fill_value=initial_mu, dtype=jnp.float64)
    rds = jnp.full(shape=(num_competitors,), fill_value=initial_rd, dtype=jnp.float64)

    # mus = jnp.array([1500.0, 1400.0, 1550.0, 1700.0])
    # rds = jnp.array([100.0, 30.0, 100.0, 300.0])

    init_val = {
        'matchups': matchups,
        'outcomes': outcomes,
        'mus': mus,
        'rds': rds,
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
            q=q,
            q2=q**2.0,
            three_q2_over_pi2=three_q2_over_pi2
        ),
        init_val=init_val,
    )
    return final_val['mus'], final_val['rds']


def run_batched_glicko(
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
    sum_1 = jnp.zeros(shape=(num_competitors,), dtype=jnp.float64)
    sum_2 = jnp.zeros(shape=(num_competitors,), dtype=jnp.float64)

    # mus = jnp.array([1500.0, 1400.0, 1550.0, 1700.0])
    # rds = jnp.array([200.0, 30.0, 100.0, 300.0])

    init_val = {
        'matchups': matchups,
        'outcomes': outcomes,
        'update_mask': update_mask,
        'mus': mus,
        'rds': rds,
        'sum_1': sum_1,
        'sum_2': sum_2,
    }
    lower = 0
    upper = matchups.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=partial(
            batched_glicko_update,
            max_rd=initial_rd,
            c2=c2,
            q=q,
            q2=q**2.0,
            three_q2_over_pi2=three_q2_over_pi2
        ),
        init_val=init_val,
    )
    return final_val['mus'], final_val['rds']

def riix_online_glicko(dataset):
    model = Glicko(
        competitors=dataset.competitors,
        update_method='iterative',
        c=0.0,
    )
    model.fit_dataset(dataset)
    return model.ratings, model.rating_devs

def main():
    dataset = load_dataset("smash_melee", '7D')
    # dataset = load_dataset("league_of_legends", '7D')

    matchups, outcomes, time_steps, update_mask = jax_preprocess(dataset)

    with timer('raax online glicko'):
        mus, rds = run_online_glicko(matchups, outcomes, num_competitors=dataset.num_competitors)
    with timer('raax online glicko'):
        mus, rds = run_online_glicko(matchups, outcomes, num_competitors=dataset.num_competitors)
    with timer('raax online glicko'):
        mus, rds = run_online_glicko(matchups, outcomes, num_competitors=dataset.num_competitors)

    with timer('raax batched glicko'):
        mus, rds = run_batched_glicko(matchups, outcomes, update_mask, num_competitors=dataset.num_competitors)
    with timer('raax batched glicko'):
        mus, rds = run_batched_glicko(matchups, outcomes, update_mask, num_competitors=dataset.num_competitors)
    with timer('raax batched glicko'):
        mus, rds = run_batched_glicko(matchups, outcomes, update_mask, num_competitors=dataset.num_competitors)

    sort_idxs = jnp.argsort(-(mus - (0.0 * rds)))
    for idx in sort_idxs[:10]:
        print(f'{dataset.competitors[idx]}: {mus[idx]:.4f}, {rds[idx]:.4f}')

    # with timer('riix online glicko'):
    #     riix_mus, riix_rds = riix_online_glicko(dataset)
    # print('riix:')
    # print(riix_mus)
    # print(riix_rds)
    # print('raax')

    # mus = jnp.array([1500.0, 1400.0, 1550.0, 1700.0], dtype=jnp.float64)
    # rds = jnp.array([100.0, 30.0, 100.0, 300.0], dtype=jnp.float64)
    # matchups = jnp.array(
    #     [[0,1],
    #      [0,2],
    #      [0,3],
    #      [2,3]]
    # )
    # outcomes = jnp.array([1.0, 0.0, 0.0, 1.0])
    # update_mask = jnp.array([False, False, False, True])
    # mus, rds = run_batched_glicko(matchups, outcomes, update_mask, num_competitors=4)

    # print(mus)
    # print(rds)

if __name__ == '__main__':
    main()