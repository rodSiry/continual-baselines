def sgd_update(key, theta, x, y, lr=1e-2):
    grad = grad_inner_loss_fn(key, theta, x, y)
    subkey, key = random.split(key, 2)
    noise = t_apply_convnet.init(key, x)
    new_theta = jax.tree_map(lambda t, g, n: t - lr * g, theta, grad, noise)
    return new_theta






def train_sgd(key, theta, x, y, n_iters=1000, bs=10):
    perm = random.permutation(key, x.shape[0])
    results = jnp.zeros(n_iters)
    transfer = []
    for i in range(n_iters):
        key, subkey = random.split(key, 2)
        test_x, test_y = get_test(key, x, y, bs)

        key, subkey = random.split(key, 2)
        batch_x, batch_y = get_test(key, x, y, bs)

        key, subkey = random.split(key, 2)
        mem_x, mem_y = get_test(key, x, y, bs)

        theta = jax.jit(sgd_update)(key, theta, batch_x, batch_y, lr=0.1)
        loss = jax.jit(acc_fn)(key, theta, test_x, test_y)
        results = results.at[i].set(loss)
        transfer.append(jax.jit(get_transfer)(key, theta, batch_x, batch_y, mem_x, mem_y))
        print(i)
    return results, transfer