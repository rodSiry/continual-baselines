import haiku as hk
import jax.numpy as jnp
from models.common import crelu
def convnet_forward(x):
    x = jnp.transpose(x, [0, 2, 3, 1])

    ly1 = hk.Conv2D(64, 2, 2)
    y = crelu(ly1(x))

    ly2 = hk.Conv2D(128, (2, 2), 2)
    y = crelu(ly2(y))

    ly3 = hk.Conv2D(256, (2, 2), 2)
    y = crelu(ly3(y))

    ly4 = hk.Conv2D(256, (2, 2), 2)
    y = crelu(ly4(y))

    y = jnp.reshape(y, (y.shape[0], -1))

    ly5 = hk.Linear(10)
    y = ly5(y)
    return y