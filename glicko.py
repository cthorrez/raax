import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import math
from functools import partial
import jax.numpy as jnp
from utils import time_function
from data_utils import get_dataset, jax_preprocess, get_synthetic_dataset

from riix.models.glicko import Glicko
from riix.utils.data_utils import MatchupDataset

# DTYPE = jnp.float16
# DTYPE = jnp.bfloat16
DTYPE = jnp.float32
# DTYPE = jnp.float64


@partial(jax.jit, static_argnums=(1,))
def g(rd_squared, three_q2_over_pi2):
    return 1.0 / jnp.sqrt(1.0 + ((rd_squared) * three_q2_over_pi2))

SIGN_FLIP = jnp.array([1.0, -1.0], dtype=DTYPE)

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

    both_outcomes = jnp.array([outcome, 1.0 - outcome], dtype=DTYPE)
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
def glicko_update(mu, rd2, sum_1, sum_2):
    mu_update_denom = (1.0 / rd2) + sum_2
    mu = mu + sum_1 / mu_update_denom
    rd2 = 1.0 / mu_update_denom
    return mu, rd2

@jax.jit
def do_update(mus, rds, fresh_rd2s, sum_1, sum_2, last_played, idx_map, idx_map_back, prev_time_step, num_competitors_seen_this_time_step):
    active_mus = mus[idx_map_back]
    new_mus, new_rd2s = glicko_update(active_mus, fresh_rd2s, sum_1, sum_2)
    new_rds = new_rd2s
    mus = mus.at[idx_map_back].set(new_mus)
    rds = rds.at[idx_map_back].set(new_rds)
    last_played = last_played.at[idx_map_back].set(prev_time_step)

    output = (
        mus,
        rds,
        jnp.zeros_like(fresh_rd2s),
        last_played,
        jnp.zeros_like(sum_1),
        jnp.zeros_like(sum_2),
        jnp.full_like(idx_map, fill_value=-1),
        jnp.full_like(idx_map_back, fill_value=-1),
        jnp.array(0, dtype=jnp.int32),
    )
    return output

@jax.jit
def do_nothing(mus, rds, fresh_rd2s, sum_1, sum_2, last_played, idx_map, idx_map_back, prev_time_step, num_competitors_seen_this_time_step):
    return mus, rds, fresh_rd2s, last_played, sum_1, sum_2, idx_map, idx_map_back, num_competitors_seen_this_time_step

@jax.jit
def seen_fn(fresh_rd2s, rds, comp_idx, idx_map, idx_map_back, last_played, cur_time_step, num_competitors_seen_this_timestep, c2, max_rd2):
    mapped_idx = idx_map[comp_idx]
    return fresh_rd2s, mapped_idx, idx_map, idx_map_back, last_played, num_competitors_seen_this_timestep

jax.jit
def not_seen_fn(fresh_rd2s, rds, comp_idx, idx_map, idx_map_back, last_played, cur_time_step, num_competitors_seen_this_timestep, c2, max_rd2):
    mapped_idx = num_competitors_seen_this_timestep
    idx_map = idx_map.at[comp_idx].set(mapped_idx)
    idx_map_back = idx_map_back.at[mapped_idx].set(comp_idx)
    cur_last_played = last_played[comp_idx]
    time_since_played = (cur_time_step - cur_last_played) * (cur_last_played != -1)
    cur_rd = rds[comp_idx]
    new_rd = jnp.minimum(cur_rd + (time_since_played * c2), max_rd2)
    fresh_rd2s = fresh_rd2s.at[mapped_idx].set(new_rd)
    num_competitors_seen_this_timestep = num_competitors_seen_this_timestep + jnp.array(1, dtype=jnp.int32)
    return fresh_rd2s, mapped_idx, idx_map, idx_map_back, last_played, num_competitors_seen_this_timestep


@partial(jax.jit, static_argnums=(2,3,4,5,6))
def batched_glicko_update(idx, prev_val, max_rd2, c2, q, q2, three_q2_over_pi2):
    mus = prev_val['mus']
    rd2s = prev_val['rd2s']
    last_played = prev_val['last_played']
    fresh_rd2s = prev_val['fresh_rd2s']
    sum_1 = prev_val['sum_1']
    sum_2 = prev_val['sum_2']
    idx_map = prev_val['idx_map']
    idx_map_back = prev_val['idx_map_back']
    num_competitors_seen_this_timestep = prev_val['num_competitors_seen_this_timestep']
    prev_time_step = prev_val['prev_time_step']
    cur_time_step = prev_val['time_steps'][idx]
    comp_idxs = prev_val['matchups'][idx]
    outcome = prev_val['outcomes'][idx]

    update_flag = cur_time_step != prev_time_step

    mus, rd2s, fresh_rd2s, last_played, sum_1, sum_2, idx_map, idx_map_back, num_competitors_seen_this_timestep = jax.lax.cond(
        update_flag,
        do_update,
        do_nothing,
        mus,
        rd2s,
        fresh_rd2s,
        sum_1,
        sum_2,
        last_played,
        idx_map,
        idx_map_back,
        prev_time_step,
        num_competitors_seen_this_timestep,
    )

    idx_1 = comp_idxs[0]
    idx_2 = comp_idxs[1]

    fresh_rd2s, mapped_idx_1, idx_map, idx_map_back, last_played, num_competitors_seen_this_timestep = jax.lax.cond(
        idx_map[idx_1] == -1,
        not_seen_fn,
        seen_fn,
        fresh_rd2s,
        rd2s,
        idx_1,
        idx_map,
        idx_map_back,
        last_played,
        cur_time_step,
        num_competitors_seen_this_timestep,
        c2,
        max_rd2,
    )
    fresh_rd2s, mapped_idx_2, idx_map, idx_map_back, last_played, num_competitors_seen_this_timestep = jax.lax.cond(
        idx_map[idx_2] == -1,
        not_seen_fn,
        seen_fn,
        fresh_rd2s,
        rd2s,
        idx_2,
        idx_map,
        idx_map_back,
        last_played,
        cur_time_step,
        num_competitors_seen_this_timestep,
        c2,
        max_rd2,
    )
    
    cur_mus = mus[comp_idxs]
    cur_rd2s = fresh_rd2s[idx_map[comp_idxs]]
    cur_gs = g(cur_rd2s, three_q2_over_pi2)

    mu_diffs = (cur_mus * SIGN_FLIP).sum() * SIGN_FLIP
    probs = jax.nn.sigmoid(q * jnp.flip(cur_gs) * mu_diffs)
     
    both_outcomes = jnp.array([outcome, 1.0 - outcome], dtype=DTYPE)
    val_1 = q * jnp.flip(cur_gs) * (both_outcomes - probs)
    sum_1 = sum_1.at[mapped_idx_1].add(val_1[0])
    sum_1 = sum_1.at[mapped_idx_2].add(val_1[1])

    val_2 = q2 * jnp.square(jnp.flip(cur_gs)) * probs * (1.0 - probs)
    sum_2 = sum_2.at[mapped_idx_1].add(val_2[0])
    sum_2 = sum_2.at[mapped_idx_2].add(val_2[1])

    new_val = {
        'matchups': prev_val['matchups'],
        'outcomes': prev_val['outcomes'],
        'time_steps': prev_val['time_steps'],
        'mus': mus,
        'rd2s': rd2s,
        'fresh_rd2s': fresh_rd2s,
        'sum_1': sum_1,
        'sum_2': sum_2,
        'last_played': last_played,
        'idx_map': idx_map,
        'idx_map_back': idx_map_back,
        'prev_time_step': cur_time_step,
        'num_competitors_seen_this_timestep': num_competitors_seen_this_timestep,
    }
    return new_val

def run_online_glicko(
    matchups,
    outcomes,
    num_competitors,
    initial_mu=1500.0,
    initial_rd=350.0,
    c=63.2,
    q=math.log(10.0)/400.0,
):
    c2 = c ** 2.0
    three_q2_over_pi2 = (3.0 * q**2.0) / (math.pi ** 2.0)
    mus = jnp.full(shape=(num_competitors,), fill_value=initial_mu, dtype=DTYPE)
    rds = jnp.full(shape=(num_competitors,), fill_value=initial_rd, dtype=DTYPE)

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
    time_steps,
    num_competitors,
    max_competitors_per_timestep,
    initial_mu=1500.0,
    initial_rd=350.0,
    c=63.2,
    q=math.log(10.0)/400.0,
):
    c2 = c ** 2.0
    q2 = q ** 2.0
    initial_rd2 = initial_rd ** 2.0
    three_q2_over_pi2 = (3.0 * q2) / (math.pi ** 2.0)
    mus = jnp.full(shape=(num_competitors + 1,), fill_value=initial_mu, dtype=DTYPE)
    rd2s = jnp.full(shape=(num_competitors + 1,), fill_value=initial_rd2, dtype=DTYPE)
    fresh_rd2s = jnp.full(shape=(max_competitors_per_timestep + 1,), fill_value=initial_rd2, dtype=DTYPE)
    sum_1 = jnp.zeros(shape=(max_competitors_per_timestep + 1,), dtype=DTYPE)
    sum_2 = jnp.zeros(shape=(max_competitors_per_timestep + 1,), dtype=DTYPE)
    last_played = jnp.full(shape=(num_competitors + 1,), fill_value=-1, dtype=jnp.int32)
    idx_map = jnp.full(shape=(num_competitors + 1,), fill_value=-1, dtype=jnp.int32)
    idx_map_back = jnp.full(shape=(max_competitors_per_timestep + 1,), fill_value=-1, dtype=jnp.int32)
   
    # mus = jnp.array([1500.0, 1400.0, 1550.0, 1700.0], dtype=DTYPE)
    # rds = jnp.array([200.0, 30.0, 100.0, 300.0], dtype=DTYPE)

    init_val = {
        'matchups': matchups,
        'outcomes': outcomes,
        'time_steps': time_steps,
        'mus': mus,
        'rd2s': rd2s,
        'fresh_rd2s': fresh_rd2s,
        'sum_1': sum_1,
        'sum_2': sum_2,
        'last_played': last_played,
        'idx_map': idx_map,
        'idx_map_back': idx_map_back,
        'prev_time_step': jnp.array(0, dtype=jnp.int32),
        'num_competitors_seen_this_timestep': jnp.array(0, dtype=jnp.int32),
    }
    lower = 0
    upper = matchups.shape[0]
    final_val = jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=partial(
            batched_glicko_update,
            max_rd2=initial_rd2,
            c2=c2,
            q=q,
            q2=q2,
            three_q2_over_pi2=three_q2_over_pi2,
        ),
        init_val=init_val,
    )
    # apply one more update to use all the remaining accumulated values
    mus, rd2s, fresh_rd2s, last_played, sum_1, sum_2, idx_map, idx_map_back, _ = do_update(
        final_val['mus'],
        final_val['rd2s'],
        final_val['fresh_rd2s'],
        final_val['sum_1'],
        final_val['sum_2'],
        final_val['last_played'],
        final_val['idx_map'],
        final_val['idx_map_back'],
        final_val['prev_time_step'],
        final_val['num_competitors_seen_this_timestep'],
    )
    time_since_played = (time_steps[-1] - last_played) * (last_played != -1)
    rds = jnp.minimum(jnp.sqrt(rd2s + (time_since_played * c2)), initial_rd)
    return mus[:-1], rds[:-1]

def run_riix_glicko(dataset, mode, initial_rating_dev=None, c=None):
    glicko = Glicko(
        competitors=dataset.competitors,
        update_method=mode,
        initial_rating_dev=initial_rating_dev,
        c=c,
    )
    glicko.fit_dataset(dataset)
    return glicko.ratings, glicko.rating_devs

def main():
    # dataset = get_synthetic_dataset(100_000, 10_000, 1_000)
    # dataset = get_dataset("smash_melee", '7D')
    # dataset = get_dataset("starcraft2", '1D')
    # dataset = get_dataset("league_of_legends", '7D')
    dataset = get_dataset('slippi', rating_period='1D')

    matchups, outcomes, time_steps, max_competitors_per_timestep = jax_preprocess(dataset)
    print(f"Max competitors per timestep: {max_competitors_per_timestep}")
    # initial_rd = 350.0
    initial_rd = 258.82
    # c = 63.2
    c = 9.19
    # c = 0.0
    n_runs = 4

    # online
    time_function(partial(run_riix_glicko, dataset, 'iterative', initial_rd, c), 'riix online glicko', n_runs)
    mus, rds = time_function(
        partial(run_online_glicko, matchups, outcomes, num_competitors=dataset.num_competitors, initial_rd=initial_rd, c=c), 'raax online glicko', n_runs
    )

    # example from pdf
    # matchups = jnp.array(
    #     [[0,1],
    #      [0,2],
    #      [0,3],
    #      [2,3]]
    # )
    # outcomes = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    # time_steps = jnp.array([0, 0, 0, 2], dtype=jnp.int32)
    # dataset = MatchupDataset.init_from_arrays(
    #     time_steps,
    #     matchups,
    #     outcomes,
    #     [0,1,2,3]
    # )
    # max_competitors_per_timestep = 4

    # batched
    riix_mus, riix_rds = time_function(partial(run_riix_glicko, dataset, 'batched', initial_rd, c), 'riix batched glicko', n_runs)

    batched_raax_glicko_func = partial(
        run_batched_glicko,
        matchups,
        outcomes,
        time_steps,
        num_competitors=dataset.num_competitors,
        max_competitors_per_timestep=max_competitors_per_timestep,
        initial_rd=initial_rd,
        c=c,
    )
    raax_mus, raax_rds = time_function(batched_raax_glicko_func, 'raax batched glicko', n_runs)
    
    top_k = 10
    rd_factor = 3.0
    riix_mus = np.asarray(riix_mus.astype(jnp.float64))
    riix_rds = np.asarray(riix_rds.astype(jnp.float64))
    riix_sort_idxs = jnp.argsort(-(riix_mus - (rd_factor * riix_rds)))
    print('riix leaderboard')
    for idx in riix_sort_idxs[:top_k]:
        print(f'{dataset.competitors[idx]}: {riix_mus[idx]:.4f}, {riix_rds[idx]:.4f}')
    print()
    
    raax_mus = np.asarray(raax_mus.astype(jnp.float64))
    raax_rds = np.asarray(raax_rds.astype(jnp.float64))
    raax_sort_idxs = jnp.argsort(-(raax_mus - (rd_factor * raax_rds)))
    print('raax leaderboard')
    for idx in raax_sort_idxs[:top_k]:
        print(f'{dataset.competitors[idx]}: {raax_mus[idx]:.4f}, {raax_rds[idx]:.4f}')

    mu_diff = np.array(jnp.abs(riix_mus - raax_mus))
    print(f'(min, mean, max) absolute mu diff: ({mu_diff.min()}, {mu_diff.mean()}, {mu_diff.max()})')
    rd_diff = jnp.abs(riix_rds - raax_rds)
    print(f'(min, mean, max) absolute rd diff: ({rd_diff.min()}, {rd_diff.mean()}, {rd_diff.max()})')

    idx = jnp.argsort(mu_diff)[-1]
    print(dataset.competitors[idx], riix_mus[idx], raax_mus[idx], riix_rds[idx], raax_rds[idx])

    import matplotlib.pyplot as plt
    abs_diffs = mu_diff

    epsilon = 1e-10
    abs_diffs_adj = abs_diffs + epsilon

    # plt.figure(figsize=(12, 8))

    # # Create histogram with linear bins
    # n, bins, patches = plt.hist(abs_diffs_adj, bins=1000, edgecolor='black')

    # plt.xscale('symlog', linthresh=1.0)  # Symmetric log scale
    # plt.yscale('symlog', linthresh=1.0)  # Symmetric log scale for y-axis too

    # plt.title('Distribution of Absolute Differences (Symlog Scale)')
    # plt.xlabel('Absolute Difference (Symlog Scale)')
    # plt.ylabel('Frequency (Symlog Scale)')
    # plt.grid(True, which="both", ls="-", alpha=0.2)

    # # Add text to point out the outliers
    # plt.text(0.95, 0.95, f"Max diff: {abs_diffs.max():.2f}", 
    #         transform=plt.gca().transAxes, ha='right', va='top')

    # # Adjust x-axis ticks
    # x_ticks = [0] + list(np.logspace(-3, np.log10(abs_diffs.max()), num=20))
    # plt.xticks(x_ticks, [f"{x:.1e}" for x in x_ticks])

    # plt.tight_layout()
    # plt.show()

    # print(riix_mus)
    # print(raax_mus)

    # # example from pdf
    # matchups = jnp.array(
    #     [[0,1],
    #      [0,2],
    #      [0,3],
    #      [2,3]]
    # )
    # outcomes = jnp.array([1.0, 0.0, 0.0])
    # time_steps = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=DTYPE)
    # mus, rds = run_batched_glicko(matchups, outcomes, time_steps, num_competitors=4, max_competitors_per_timestep=3)
    # print(mus)
    # print(rds)

if __name__ == '__main__':
    main()