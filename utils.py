import dill as pickle
import numpy as np
import jax.numpy as jnp
import jax


## directly from https://github.com/milutter/deep_lagrangian_networks/blob/main/deep_lagrangian_networks/utils.py
def init_env(args):
    # Set the NumPy Formatter:
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    # Set the seed:
    np.random.seed(seed)
    return seed,  render, load_model, save_model


def load_dataset(n_characters=3, filename="data/character_data.pickle", test_label=("e", "q", "v")):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    n_dof = 2

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    # test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean


class ReplayMemory:
    def __init__(self, maximum_number_of_samples, minibatch_size, dim):

        # General Parameters:
        # self._max_samples: 总数据量
        self._max_samples = maximum_number_of_samples
        # self._minibatch_size： 一个 batch 数据量
        self._minibatch_size = minibatch_size
        # self._dim => 数据中不同量的维度: q and qd should have the same dimension, but tau can have a different dimension
        self._dim = dim

        self._data_idx = 0
        self._data_n = 0

        # Sampling:
        self._sampler_idx = 0
        self._order = None

        # Data Structure:
        # create an empty list whose length is the number of data kind (if (q, qd, tau, q_next, qd_next), then 5)
        # evey array in that list, its shape will be the dimension of that kind of data and the number of data
        self._data = []
        for i in range(len(dim)):
            self._data.append(np.empty((self._max_samples, ) + dim[i]))

    def __iter__(self):
        # Shuffle data and reset counter:
        self._order = np.random.permutation(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]

        # Reject Batches that have less samples:
        if batch_idx.size < self._minibatch_size:
            raise StopIteration()

        out = [x[batch_idx] for x in self._data]
        return out

    def add_samples(self, data):
        assert len(data) == len(self._data)

        # Add samples:
        add_idx = self._data_idx + np.arange(data[0].shape[0])
        add_idx = np.mod(add_idx, self._max_samples)
        # in normal case, add_idx should be a list from 0 to self._max_samples -1
        # [0, 1, ...,  self._max_samples-1]

        for i in range(len(data)):
            self._data[i][add_idx] = data[i][:]
            # print(f"adding {i + 1} data to the self._data")

        # Update index:
        self._data_idx = np.mod(add_idx[-1] + 1, self._max_samples)
        #  self._data_idx  will be [0, 1, ...,  self._max_samples-1]
        self._data_n = min(self._data_n + data[0].shape[0], self._max_samples)
        # self._data_n will be self._max_samples

        # Clear excessive GPU Memory:
        del data

    def shuffle(self):
        self._order = np.random.permutation(self._data_idx)
        self._sampler_idx = 0

    def get_full_mem(self):
        out = [x[:self._data_n] for x in self._data]
        return out

    def not_empty(self):
        return self._data_n > 0
