import haiku as hk
import jax.numpy as jnp
from models.common import crelu
import jax.nn as nn
def gated_convnet_forward(x):
    x = jnp.transpose(x, [0, 2, 3, 1])

    ly1 = hk.Conv2D(64, 2, 2)
    ft1 = hk.Conv2D(64, 2, 2)
    y = nn.relu(ly1(x)) * nn.sigmoid(ft1(x))

    ly2 = hk.Conv2D(128, (2, 2), 2)
    ft2 = hk.Conv2D(128, (2, 2), 2)
    y = nn.relu(ly2(y)) * nn.sigmoid(ft2(y))

    ly3 = hk.Conv2D(256, (2, 2), 2)
    ft3 = hk.Conv2D(256, (2, 2), 2)
    y = nn.relu(ly3(y)) * nn.sigmoid(ft3(y))

    ly4 = hk.Conv2D(256, (2, 2), 2)
    ft4 = hk.Conv2D(256, (2, 2), 2)
    y = nn.relu(ly4(y)) * nn.sigmoid(ft4(y))

    y = jnp.reshape(y, (y.shape[0], -1))

    ly5 = hk.Linear(100)
    y = ly5(y)
    return y