from training.base import Trainer
import jax.numpy as jnp
import jax.random as random
from buffers.buffers import get_replay, get_test, get_batch
import optax
import tqdm
import functools
import jax

class AdamTrainer(Trainer):
    def __init__(self, key, model, criterion, n_iters, bs, lr, logger=None):
        super().__init__(key, model, criterion, n_iters, bs, logger=logger)
        self.lr = lr

    def fit(self, key, x, y):
        n_repeat = 1

        self.theta = self.init_theta(key, x)
        accuracies = jnp.zeros(self.n_iters)
        transfers = jnp.zeros(self.n_iters)
        optim = optax.adam(self.lr)
        opt_state = optim.init(self.theta)
        batch_index = 0
        for iter_index in tqdm.tqdm(range(self.n_iters // n_repeat)):

            key, subkey = random.split(key, 2)
            test_x, test_y = get_test(key, x, y, self.bs)

            key, subkey = random.split(key, 2)
            mem_x, mem_y = get_test(key, x, y, self.bs)

            batch_x, batch_y = get_batch(key, x, y, self.bs, batch_index)

            grad = self.grad_loss_function(key, self.theta, batch_x, batch_y)
            updates, opt_state = optim.update(grad, opt_state, self.theta)
            self.theta = optax.apply_updates(updates, self.theta)
            self.logger.track(self.acc_function(key, self.theta, test_x, test_y), name='accuracy', context={"algo":"adam"}, step=iter_index)
            self.logger.track(self.get_transfer(key, self.theta, batch_x, batch_y, mem_x, mem_y), name='transfer', context={"algo":"adam"} , step=iter_index)
            batch_index += 1
