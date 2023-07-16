import functools

import jax
import jax.numpy as jnp
import jax.random as random
from training.base import Trainer
from buffers.buffers import get_replay, get_test, get_batch
from models.utils import tree_dot_product, clip_grad_norm
import functools



class MERTrainer(Trainer):
    def __init__(self, key, model, criterion, n_iters, bs):
        super().__init__(key, model, criterion, n_iters, bs)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def mer_update(self, key, theta, x1, y1, x2, y2, lr=1e-2, reg=0.01):
        def mer_objective(key, theta, x1, y1, x2, y2):
            value1, grad1 = jax.value_and_grad(self.loss_function, argnums=1)(key, theta, x1, y1)
            value2, grad2 = jax.value_and_grad(self.loss_function, argnums=1)(key, theta, x2, y2)
            loss = 0 * (value1 + value2) - reg * tree_dot_product(grad1, grad2)
            return loss

        subkey, key = random.split(key, 2)
        grad = jax.grad(mer_objective, argnums=1)(key, theta, x1, y1, x2, y2)
        return jax.tree_map(lambda t, g: t - lr * g, theta, grad)

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
            mem_x, mem_y = get_replay(key, x, y, self.bs, batch_index)

            batch_x, batch_y = get_batch(key, x, y, self.bs, batch_index)

            self.theta = self.mer_update(key, self.theta, batch_x, batch_y, mem_x, mem_y, lr=0.1)

            accuracy = self.acc_function(key, self.theta, test_x, test_y)
            accuracies = accuracies.at[iter_index].set(accuracy)
            print(iter_index)
            transfers = transfers.at[iter_index].set(self.get_transfer(key, self.theta, batch_x, batch_y, mem_x, mem_y))
            batch_index += 1
            iter_index += 1
            if iter_index == self.n_iters:
                break
        return accuracies, transfers




"""
def mer_update(key, theta, x1, y1, x2, y2, lr=1e-2, reg=0.01):
    def mer_objective(key, theta, x1, y1, x2, y2):
        value1, grad1 = jax.value_and_grad(loss_fn, argnums=1)(key, theta, x1, y1)
        value2, grad2 = jax.value_and_grad(loss_fn, argnums=1)(key, theta, x2, y2)
        loss = 0.5 * (value1 + value2) - reg * tree_dot_product(clip_grad_norm(grad1), clip_grad_norm(grad2))
        return loss

    subkey, key = random.split(key, 2)
    grad = jax.grad(mer_objective, argnums=1)(key, theta, x1, y1, x2, y2)
    new_theta = jax.tree_map(lambda t, g: t - lr * g, theta, grad)
    return new_theta

def train_mer(key, theta, x, y, n_iters=1000, bs=10, threshold=0.01):
    results = jnp.zeros(n_iters)
    batch_indices = []
    transfer = []
    iter_index = 0
    batch_index = 0

    while True:
        key, subkey = random.split(key, 2)
        test_x, test_y = get_test(key, x, y, bs)

        key, subkey = random.split(key, 2)
        mem_x, mem_y = get_replay(key, x, y, bs, batch_index)

        batch_x, batch_y = get_batch(key, x, y, bs, batch_index)

        theta = jax.jit(mer_update)(key, theta, batch_x, batch_y, mem_x, mem_y, lr=0.1)
        loss = jax.jit(acc_fn)(key, theta, test_x, test_y)
        results = results.at[iter_index].set(loss)
        print(iter_index)
        batch_indices.append(batch_index)
        transfer.append(jax.jit(get_transfer)(key, theta, batch_x, batch_y, mem_x, mem_y))
        batch_index += 1
        iter_index += 1
        if iter_index == n_iters:
            break
    return results, transfer
"""