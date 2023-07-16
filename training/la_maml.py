import jax.nn as nn
import jax

def transform_param(x):
    return nn.relu(x)

def sgd_meta_update(key, theta, phi, x, y):
    grad = grad_inner_loss_fn(key, theta, x, y)
    new_theta = jax.tree_map(lambda t, p, g: t - transform_param(p) * g, theta, phi, grad)
    return new_theta

def la_maml_inner_unroll(key, theta, phi, x, y, x_test, y_test):
    def inner_fun(acc, input_data):
        acc = sgd_meta_update(key, acc, phi, jnp.expand_dims(input_data["x"], 0), jnp.expand_dims(input_data["y"], 0))
        return acc, 0
    input_data = {'x':x, 'y':y}
    theta, _ = jax.lax.scan(inner_fun, theta, input_data)
    meta_loss = loss_fn(key, theta, x_test, y_test)
    return meta_loss

def la_maml_update(key, theta, phi, x, y, x_test, y_test, outer_lr=1e-2):
    phi_grad = jax.grad(la_maml_inner_unroll, argnums=2)(key, theta, phi, x, y, x_test, y_test)
    theta_grad = jax.grad(la_maml_inner_unroll, argnums=1)(key, theta, phi, x, y, x_test, y_test)
    new_phi = jax.tree_map(lambda p, g: p - outer_lr * g, phi, phi_grad)
    new_theta = jax.tree_map(lambda t, p, g: t - transform_param(p) * g, theta, new_phi, theta_grad)
    return new_theta, new_phi


def train_la_maml(key, theta, x, y, n_iters=1000, bs=10, threshold=0.01, initial_phi=0.01):

    phi = jax.tree_map(lambda p: jnp.ones(p.shape) * initial_phi, theta)
    results = jnp.zeros(n_iters)
    batch_indices = []
    iter_index = 0
    batch_index = 0

    while True:
        key, subkey = random.split(key, 2)
        test_x, test_y = get_test(key, x, y, bs)

        key, subkey = random.split(key, 2)
        mem_x, mem_y = get_replay(key, x, y, bs, batch_index)

        batch_x, batch_y = get_batch(key, x, y, bs, batch_index)
        val_x, val_y = jnp.concatenate([batch_x, mem_x], 0), jnp.concatenate([batch_y, mem_y], 0)

        theta, phi = jax.jit(la_maml_update)(key, theta, phi, batch_x, batch_y, val_x, val_y, outer_lr=0.01)
        loss = jax.jit(loss_fn)(key, theta, test_x, test_y)

        val_loss = jax.jit(loss_fn)(key, theta, val_x, val_y)
        results = results.at[iter_index].set(loss)

        iter_index += 1
        if iter_index == n_iters:
            break
        if val_loss < threshold:
            batch_index += 1
        print(iter_index, get_phi_stats(phi))
        batch_indices.append(batch_index)
    return results, batch_indices