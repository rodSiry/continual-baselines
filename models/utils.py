import jax
import functools
import jax.numpy as jnp
def tree_dot_product(t1, t2):
    prod = jax.tree_map(lambda x, y: jnp.array([(x * y).sum(), (x * x).sum(), (y * y).sum()]), t1, t2)
    norm = functools.reduce(lambda x, y: x + y, jax.tree_util.tree_leaves(prod), jnp.zeros((3,)))
    return norm[0] / jnp.sqrt(norm[1]) / jnp.sqrt(norm[2])

def clip_grad_norm(g, clip_value=1):
    prod = jax.tree_map(lambda x: (x * x).sum(), g)
    norm = functools.reduce(lambda x, y: x + y, jax.tree_util.tree_leaves(prod), 0)
    coeff = jnp.minimum(norm, clip_value) / norm
    norm_grad = jax.tree_map(lambda x:  coeff * x, g)
    return norm_grad

def get_tree_stats(phi):
    min_val = functools.reduce(lambda x, y: min(x, jnp.min(y)), jax.tree_leaves(phi), 1000)
    max_val = functools.reduce(lambda x, y: max(x, jnp.max(y)), jax.tree_leaves(phi), 0)
    mean_val = functools.reduce(lambda x, y: (x[0] + y.sum(), x[1] + y.size), jax.tree_leaves(phi), (0, 0))
    return min_val.item(), max_val.item(), mean_val[0].item() / mean_val[1]