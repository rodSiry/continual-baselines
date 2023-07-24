import functools

import jax
import jax.numpy as jnp
import jax.random as random
from training.base import Trainer
from buffers.buffers import get_replay, get_test, get_batch
from models.utils import tree_dot_product, clip_grad_norm
import functools
import optax
import tqdm

class MERTrainer(Trainer):
    def __init__(self, key, model, criterion, n_iters, bs, lr, logger=None):
        super().__init__(key, model, criterion, n_iters, bs, logger=logger)
        self.lr = lr

    @functools.partial(jax.jit, static_argnums=(0, ))
    def mer_update(self, key, theta, x1, y1, x2, y2, lr=1e-2, reg=0):
        def mer_objective(key, theta, x1, y1, x2, y2):
            value1, grad1 = jax.value_and_grad(self.loss_function, argnums=1)(key, theta, x1, y1)
            value2, grad2 = jax.value_and_grad(self.loss_function, argnums=1)(key, theta, x2, y2)
            loss = 0.5 * (value1 + value2) - reg * tree_dot_product(grad1, grad2)
            return loss

        subkey, key = random.split(key, 2)
        grad = jax.grad(mer_objective, argnums=1)(key, theta, x1, y1, x2, y2)
        return grad


    def fit(self, key, x, y):
        task_size=1000
        n_tasks = x.shape[0] // task_size
        print(n_tasks)
        self.theta = self.init_theta(key, x)
        accuracies = jnp.zeros(self.n_iters*n_tasks)
        in_task_accuracies = jnp.zeros(self.n_iters*n_tasks)
        transfers = jnp.zeros(self.n_iters*n_tasks)
        optim = optax.adam(self.lr)
        opt_state = optim.init(self.theta)

        for task_index in range(n_tasks):
            task_x, task_y = get_batch(key, x, y, task_size, task_index)

            for iter_index in tqdm.tqdm(range(self.n_iters)):
                key, subkey = random.split(key, 2)
                test_x, test_y = get_test(key, x, y, self.bs)

                key, subkey = random.split(key, 2)
                mem_x, mem_y = get_test(key, x, y, self.bs)

                batch_x, batch_y = get_batch(key, task_x, task_y, self.bs, iter_index)
                in_task_accuracy = self.acc_function(key, self.theta, batch_x, batch_y)

                grad = self.mer_update(key, self.theta, batch_x, batch_y, mem_x, mem_y, lr=0.1)
                updates, opt_state = optim.update(grad, opt_state, self.theta)
                self.theta = optax.apply_updates(updates, self.theta)
                accuracy = self.acc_function(key, self.theta, test_x, test_y)

                index = task_index*self.n_iters + iter_index
                self.logger.track(accuracy, name='accuracy', context={"algo":"mer"}, step=index)
                self.logger.track(in_task_accuracy, name='accuracy', context={"algo":"mer", "subset":"in_task"}, step=index)
                self.logger.track(self.get_transfer(key, self.theta, batch_x, batch_y, mem_x, mem_y), context={"algo":"mer"}, name='transfer', step=index)