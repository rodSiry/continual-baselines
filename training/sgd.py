from training.base import Trainer
import jax.numpy as jnp
import jax.random as random
from buffers.buffers import get_replay, get_test, get_batch
import optax
import jax

class SGDTrainer(Trainer):
    def __init__(self, key, model, criterion, n_iters, bs):
        super().__init__(key, model, criterion, n_iters, bs)

    def fit(self, key, x, y):
        self.theta = self.init_theta(key, x)
        accuracies = jnp.zeros(self.n_iters)
        transfers = jnp.zeros(self.n_iters)
        iter_index = 0
        batch_index = 0
        while True:
            key, subkey = random.split(key, 2)
            test_x, test_y = get_test(key, x, y, self.bs)

            key, subkey = random.split(key, 2)
            mem_x, mem_y = get_test(key, x, y, self.bs)

            batch_x, batch_y = get_batch(key, x, y, self.bs, batch_index)

            grad = self.grad_loss_function(key, self.theta, batch_x, batch_y)
            self.theta = jax.tree_util.tree_map(lambda t, g: t - 0.1 * g, self.theta, grad)

            accuracy = self.acc_function(key, self.theta, test_x, test_y)
            accuracies = accuracies.at[iter_index].set(accuracy)
            print(iter_index)
            transfers = transfers.at[iter_index].set(self.get_transfer(key, self.theta, batch_x, batch_y, mem_x, mem_y))
            batch_index += 1
            iter_index += 1
            if iter_index == self.n_iters:
                break
        return accuracies, transfers