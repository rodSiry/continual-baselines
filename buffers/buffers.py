import jax.numpy as jnp
import jax.random as random

def get_replay(key, x, y, bs, batch_index):
    max_val = min(bs * batch_index + bs, x.shape[0] - 1)
    indices = random.randint(key, (bs, ), 0, max_val)
    return x[indices], y[indices]

def get_test(key, x, y, bs):
    indices = random.randint(key, (bs, ), 0, x.shape[0] - 1)
    return  x[indices], y[indices]

def get_batch(key, x, y, bs, batch_index):
    return jnp.roll(x, bs * batch_index, axis=0)[:bs], jnp.roll(y, bs * batch_index, axis=0)[:bs]