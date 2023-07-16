import haiku as hk
import jax.numpy as jnp
from models.common import crelu
import jax.nn as nn


def residual_layer(out, upsampling='down'):
    short = hk.Conv2D(out, (1, 1))
    first = hk.Conv2D(out, (3, 3), padding=(1, 1))
    second = hk.Conv2D(out, (3, 3), padding=(1, 1))

    if upsampling == 'same':
        last = hk.Conv2D(out)
    elif upsampling == 'down':
        last = hk.Conv2D(out, (2, 2), stride=2)
    elif upsampling == 'up':
        last = hk.Conv2DTranspose(out, (2, 2), stride=2)

    return lambda x : last(short(x) + nn.relu(second(nn.relu(first(x)))))

def resnet_forward(x):
    x = jnp.transpose(x, [0, 2, 3, 1])

    ly1 = residual_layer(32)
    y = crelu(ly1(x))

    ly2 = residual_layer(64)
    y = crelu(ly2(y))

    ly3 = residual_layer(128)
    y = crelu(ly3(y))

    ly4 = residual_layer(128)
    y = crelu(ly4(y))

    y = jnp.reshape(y, (y.shape[0], -1))

    ly5 = hk.Linear(10)
    y = ly5(y)
    return y