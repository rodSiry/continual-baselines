def train_progressive_sgd(key, theta, x, y, n_iters=1000, bs=10, threshold=0.01):
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
        batch_x, batch_y = jnp.concatenate([batch_x, mem_x], 0), jnp.concatenate([batch_y, mem_y], 0)

        theta = jax.jit(sgd_update)(key, theta, batch_x, batch_y, lr=0.1)
        loss = loss_fn(key, theta, test_x, test_y)
        val_loss = loss_fn(key, theta, batch_x, batch_y)
        results = results.at[iter_index].set(loss)

        iter_index += 1
        if iter_index == n_iters:
            break
        if val_loss < threshold:
            batch_index += 1
        print(iter_index)
        batch_indices.append(batch_index)
    return results, batch_indices