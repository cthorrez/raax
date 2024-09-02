import jax.numpy as jnp
import multiprocessing as mp
from itertools import product
from clayto import Elo, Clayto

def eval_func(cls, matchups, outcomes, num_competitors, model_params):
    model = cls(num_competitors=num_competitors **model_params)
    probs, _ = model.run(matchups, outcomes)

def sweep(cls, matchups, outcomes, num_competitors, params):
    pool = mp.Pool(12)
    model = cls(num_competitors=num_competitors **params)


def main():
    N = 10
    scales = jnp.linspace(start = 10.0, stop=500.0, num=N)
    ks = jnp.linspace(start = 1.0, stop = 100.0, num=N)
    elo_params = {'scale' : scales, 'k':ks}
    elo_params = [dict(zip(elo_params.keys(), values)) for values in product(*elo_params.values())]
    


if __name__ == '__main__':
    main()