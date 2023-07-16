from models.common import one_hot
import jax.numpy as jnp
def l2_one_hot(y_, y):
    y = one_hot(y, 10)
    loss = ((y_ - y) ** 2).mean()
    return loss

def acc_fn(y_, y):
    y_ = jnp.argmax(y_, -1)
    loss = (y == y_).astype(int).mean()
    return loss
