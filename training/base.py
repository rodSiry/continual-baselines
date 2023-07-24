import haiku as hk
from models.losses import acc_fn
from models.utils import tree_dot_product
import jax
import functools
class Trainer:
    def __init__(self, key, model, criterion, n_iters, bs, logger=None):
        self.key = key
        self.model = hk.transform(model)
        self.criterion = criterion
        self.n_iters = n_iters
        self.bs = bs
        self.logger = logger

    @functools.partial(jax.jit, static_argnums=(0,))
    def init_theta(self, key, x):
        theta = self.model.init(key, x)
        return theta

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_function(self, key, theta, x, y):
        y_ = self.model.apply(theta, key, x)
        return self.criterion(y_, y)

    @functools.partial(jax.jit, static_argnums=(0,))
    def grad_loss_function(self, key, theta, x, y):
        return jax.grad(self.loss_function, argnums=1)(key, theta, x, y)

    @functools.partial(jax.jit, static_argnums=(0,))
    def acc_function(self, key, theta, x, y):
        y_ = self.model.apply(theta, key, x)
        return acc_fn(y_, y)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_transfer(self, key, theta, x1, y1, x2, y2):
        grad1 = jax.grad(self.loss_function, argnums=1)(key, theta, x1, y1)
        grad2 = jax.grad(self.loss_function, argnums=1)(key, theta, x2, y2)
        return tree_dot_product(grad1, grad2)

