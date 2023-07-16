import jax.numpy as jnp
import jax
import jax.nn as nn

def normalize(x):
    mean = jnp.expand_dims(jnp.mean(x, 0), 0)
    std = jnp.expand_dims(jnp.std(x, 0), 0)
    return (x - mean) / (std + 1e-10)

def crelu(x):
    return jnp.concatenate([nn.relu(x), nn.relu(-x)], -1)

def one_hot(x, N):
    x = x.astype(int)
    result = jnp.zeros((x.shape[0], N))
    result = jax.vmap(lambda x, i: x.at[i].set(1))(result, x)
    return result