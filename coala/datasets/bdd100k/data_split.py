import numpy as np
import logging

logger = logging.getLogger(__name__)


def shuffle(data):
    num_of_data = len(data)
    data = np.array(data)
    index = [i for i in range(num_of_data)]
    np.random.shuffle(index)
    data = data[index]
    return data


def get_attribute_priors(data_y):
    """
    Args:
        data_y (array[int]): The array of one-hot label data.

    Returns:
        array[float]: the array if prior distribution
    """
    data_num = len(data_y)
    label_num = len(np.unique(data_y))
    distribution = [0 for _ in range(label_num)]
    unique_values, counts = np.unique(data_y, return_counts=True)
    for i in range(len(unique_values)):
        distribution[unique_values[i]] = counts[i] / data_num
    return np.array(distribution)


def equal_division(num_groups, data):
    """Partition data into multiple clients with equal quantity.

    Args:
        num_groups (int): THe number of groups to partition to.
        data (list[Object]): A list of elements to be divided.

    Returns:
        list[list]: A list where each element is a list of data of a group/client.
    """
    np.random.shuffle(data)
    num_of_data = len(data)
    assert num_of_data > 0
    data_per_client = num_of_data // num_groups
    large_group_num = num_of_data - num_groups * data_per_client
    small_group_num = num_groups - large_group_num
    split_data = []
    for i in range(small_group_num):
        base_index = data_per_client * i
        split_data.append(data[base_index: base_index + data_per_client])
    small_size = data_per_client * small_group_num
    data_per_client += 1
    for i in range(large_group_num):
        base_index = small_size + data_per_client * i
        split_data.append(data[base_index: base_index + data_per_client])

    return split_data


def iid(data_x, data_y, num_of_clients):
    """Partition dataset into multiple clients with equal data quantity (difference is less than 1) randomly.

    Args:
        data_x (array[int]): A array of data.
        data_y (array[int]): A array of data.
        num_of_clients (int): The number of clients to partition to.

    Returns:
        list[str]: A list of per-client image ids.
    """
    data = shuffle(data_x)
    divided_list = equal_division(num_of_clients, data)
    federated_data = []
    for i in range(num_of_clients):
        temp_client = np.array(divided_list[i]).astype(int).tolist()
        federated_data.append(temp_client)

    return federated_data


def non_iid_dirichlet(data_x, data_y, num_of_clients, alpha, min_size):
    """Partition dataset into multiple clients following the Dirichlet process.

    Args:
        data_x (array[int]): A list of data.
        data_y (array[int]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        alpha (float): The parameter for Dirichlet process simulation.
        min_size (int): The minimum number of data size of a client.

    Returns:
        list: A list of per-client image ids.
    """
    current_min_size = 0
    num_class = np.amax(data_y) + 1
    data_size = len(data_y)

    while current_min_size < min_size:
        idx_batch = [[] for _ in range(num_of_clients)]
        for k in range(num_class):
            idx_k = np.where(data_y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_of_clients))
            # using the proportions from dirichlet, only select those clients having data amount less than average
            proportions = np.array(
                [p * (len(idx_j) < data_size / num_of_clients) for p, idx_j in zip(proportions, idx_batch)])
            # scale proportions
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            current_min_size = min([len(idx_j) for idx_j in idx_batch])

    federated_data = []
    for j in range(num_of_clients):
        np.random.shuffle(idx_batch[j])
        temp = np.array(data_x[idx_batch[j]]).astype(int).tolist()
        federated_data.append(temp)

    return federated_data


def non_iid_dirichlet_hierarchical(data_x, data_y, num_of_clients, alpha1=1.0, alpha2=1.0, alpha3=1.0, min_size=10):
    """Partition dataset into multiple clients following the Dirichlet process.

    Args:
        data_x (array[int]): A list of data index.
        data_y (array[int]): A list of data attributes (labels).
        num_of_clients (int): The number of clients to partition to.
        alpha1 (float): The parameter for Dirichlet process simulation.
        alpha2 (float): The parameter for Dirichlet process simulation.
        alpha3 (float): The parameter for Dirichlet process simulation.
        min_size (int): The minimum number of data size of a client.

    Returns:
        list: A list of per-client image ids.
    """
    current_min_size = 0
    num_class1 = np.amax(data_y[:, 0]) + 1
    num_class2 = np.amax(data_y[:, 1]) + 1
    num_class3 = np.amax(data_y[:, 2]) + 1
    data_size = len(data_y)

    data_y1 = data_y[:, 0]
    data_y2 = data_y[:, 1]
    data_y3 = data_y[:, 2]

    # prior computation
    py2 = get_attribute_priors(data_y2)
    py3 = get_attribute_priors(data_y3)
    alphas_2 = alpha2 * py2
    alphas_3 = alpha3 * py3

    while current_min_size < min_size:
        idx_batch = [[] for _ in range(num_of_clients)]
        for k in range(num_class1):
            idx_k = np.where(data_y1 == k)[0]
            proportions1 = np.random.dirichlet(np.repeat(alpha1, num_of_clients))
            # using the proportions from dirichlet, only select those clients having data amount less than average
            proportions1 = np.array(
                [p * (len(idx_j) < data_size / num_of_clients) for p, idx_j in zip(proportions1, idx_batch)])
            # scale proportions
            proportions1 = proportions1 / proportions1.sum()
            proportions2 = np.random.dirichlet(alphas_2, num_of_clients)
            proportions3 = np.random.dirichlet(alphas_3, num_of_clients)
            for p in range(num_class2):
                for q in range(num_class3):
                    idx_kp = np.intersect1d(np.where(data_y1 == k)[0], np.where(data_y2 == p)[0])
                    idx_kpq = np.intersect1d(idx_kp, np.where(data_y3 == q)[0])
                    for cid in range(num_of_clients):
                        data_len = int(proportions3[cid][q] * proportions2[cid][p] * proportions1[cid] * len(idx_k))
                        data_len_kpq = min(data_len, len(idx_kpq))
                        if data_len_kpq > 0:
                            idx_batch[cid] = idx_batch[cid] + np.random.choice(idx_kpq, data_len_kpq,
                                                                               replace=False).tolist()
        current_min_size = min([len(idx_j) for idx_j in idx_batch])

    federated_data = []
    for j in range(num_of_clients):
        np.random.shuffle(idx_batch[j])
        temp = np.array(data_x[idx_batch[j]]).astype(int).tolist()
        federated_data.append(temp)

    return federated_data
