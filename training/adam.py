from training.base import Trainer
import jax.numpy as jnp
import jax.random as random
from buffers.buffers import get_replay, get_test, get_batch
import optax

class AdamTrainer(Trainer):
    def __init__(self, key, model, criterion, n_iters, bs):
        super().__init__(key, model, criterion, n_iters, bs)

    def fit(self, key, x, y):
        self.theta = self.init_theta(key, x)
        accuracies = jnp.zeros(self.n_iters)
        transfers = jnp.zeros(self.n_iters)
        optim = optax.adam(1e-4)
        opt_state = optim.init(self.theta)
        iter_index = 0
        batch_index = 0
        while True:
            key, subkey = random.split(key, 2)
            test_x, test_y = get_test(key, x, y, self.bs)

            key, subkey = random.split(key, 2)
            mem_x, mem_y = get_replay(key, x, y, self.bs, batch_index)

            batch_x, batch_y = get_batch(key, x, y, self.bs, batch_index)

            grad = self.grad_loss_function(key, self.theta, batch_x, batch_y)
            updates, opt_state = optim.update(grad, opt_state, self.theta)
            self.theta = optax.apply_updates(updates, self.theta)

            accuracy = self.acc_function(key, self.theta, test_x, test_y)
            accuracies = accuracies.at[iter_index].set(accuracy)
            print(iter_index)
            transfers = transfers.at[iter_index].set(self.get_transfer(key, self.theta, batch_x, batch_y, mem_x, mem_y))
            batch_index += 1
            iter_index += 1
            if iter_index == self.n_iters:
                break
        return accuracies, transfers

def train_adam(key, theta, x, y, n_iters=1000, bs=10):
    results = jnp.zeros(n_iters)
    optim = optax.adam(1e-4)
    opt_state = optim.init(theta)
    transfer = []
    batch_index = 0

    for i in range(n_iters):
        key, subkey = random.split(key, 2)
        test_x, test_y = get_test(key, x, y, bs)

        key, subkey = random.split(key, 2)
        mem_x, mem_y = get_replay(key, x, y, bs, batch_index)

        batch_x, batch_y = get_batch(key, x, y, bs, batch_index)

        grad = jax.jit(grad_inner_loss_fn)(key, theta, batch_x, batch_y)
        updates, opt_state = optim.update(grad, opt_state, theta)
        theta = optax.apply_updates(updates, theta)

        transfer.append(jax.jit(get_transfer)(key, theta, batch_x, batch_y, mem_x, mem_y))

        loss = jax.jit(acc_fn)(key, theta, test_x, test_y)
        results = results.at[i].set(loss)
        print(i)
        batch_index += 1
    return results, transfer