import jax.random as random
from training.mer import MERTrainer
from training.adam import AdamTrainer
from models.losses import l2_one_hot
from models.convnet import convnet_forward
from models.resnet import resnet_forward
from loaders import SequenceGenerator
from matplotlib import pyplot as plt
from utils import convolve

n_iters = 1000
bs = 10
seq_len = 10000
conv_value = 10
data = SequenceGenerator()
x, y = data.gen_sequence(seq_len)

key = random.PRNGKey(10)

trainer = AdamTrainer(key, convnet_forward, l2_one_hot, n_iters, bs)
accuracies, transfers = trainer.fit(key, x, y)

trainer = MERTrainer(key, convnet_forward, l2_one_hot, n_iters, bs)
mer_accuracies, mer_transfers = trainer.fit(key, x, y)

plt.subplot(2, 1, 1)
plt.plot(convolve(accuracies, conv_value))
plt.plot(convolve(mer_accuracies, conv_value))

plt.subplot(2, 1, 2)
plt.plot(convolve(transfers, conv_value))
plt.plot(convolve(mer_transfers, conv_value))

plt.savefig("new_graph.png")



