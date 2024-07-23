import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


def data_distribution_display(data_y, data_index_map, show=False, save=True, save_path=None):
    attributes = {
        0: "weather",
        1: "scene",
        2: "timeofday"
    }
    for y_dim in range(data_y.shape[1]):
        print_data_distribution(data_y[:, y_dim], data_index_map, attributes[y_dim], show, save, save_path)


def print_data_distribution(data_y, data_index_map, attribute, show=False, save=True, save_path=None):
    """Log the distribution of client datasets."""
    data_distribution = {}
    label_num = len(np.unique(data_y))
    for cid, data_idx in enumerate(data_index_map):
        distribution = [0 for _ in range(label_num)]
        unique_values, counts = np.unique(data_y[data_idx], return_counts=True)
        for i in range(len(unique_values)):
            distribution[unique_values[i]] = counts[i]
        data_distribution[cid] = np.array(distribution)
    logger.info(data_distribution)

    # print

    categories = {
        "weather": ["rainy", "snowy", "clear", "overcast", "partly cloudy", "foggy"],
        "scene": ["tunnel", "residential", "parking lot", "city street", "gas stations", "highway"],
        "timeofday": ["daytime", "night", "dawn/dusk"]
    }
    client_num = len(data_distribution)
    client_idx = [i + 1 for i in range(client_num)]
    category_idx = [i + 1 for i in range(label_num)]
    plt.figure(figsize=(30, 6))
    for c in range(client_num):
        cid = [c + 1] * label_num
        plt.scatter(x=cid, y=category_idx, s=data_distribution[c] * 10, c='blue', alpha=1.0)
    plt.xticks([r for r in client_idx], client_idx, fontsize=6)
    plt.yticks([r for r in category_idx], categories[attribute], fontsize=20)

    plt.xlim([0, client_num + 1])
    plt.ylim([0, label_num + 1])
    plt.xlabel(u'Client Id', fontsize=20)
    plt.ylabel(u'Class Id (%)', fontsize=20)
    if save:
        file_path = "data/bdd100k" if save_path is None else save_path
        plt.savefig(os.path.join(file_path, "{}_dist.pdf".format(attribute)))
    if show:
        plt.show()


def datasize_distribution(data_index_map, show=False, save=True, save_path=None):
    num_client = len(data_index_map)
    size_list = []
    for i in range(num_client):
        size_list.append(len(data_index_map[i]))
    plt.figure(figsize=(8, 6))
    sns.histplot(size_list, kde=True,
                 bins=20, color='red',
                 line_kws={'linewidth': 2},
                 )
    if save:
        file_path = "data/bdd100k" if save_path is None else save_path
        plt.savefig(os.path.join(file_path, "bdd_datasize.pdf"))
    if show:
        plt.show()


def pairwise_divergence(data_y, data_index_map, show=False, save=True, save_path=None):
    # Jensen–Shannon divergence
    # average over three attributes
    data_y1 = data_y[:, 0]
    data_y2 = data_y[:, 1]
    data_y3 = data_y[:, 2]

    data_distribution1 = get_label_distribution(data_y1, data_index_map)
    data_distribution2 = get_label_distribution(data_y2, data_index_map)
    data_distribution3 = get_label_distribution(data_y3, data_index_map)

    jsd1 = get_pairwise_js_divergence(data_distribution1)
    jsd2 = get_pairwise_js_divergence(data_distribution2)
    jsd3 = get_pairwise_js_divergence(data_distribution3)

    jsd = (jsd1 + jsd2 + jsd3) / 3

    plt.figure(figsize=(8, 6))
    sns.histplot(jsd, kde=True,
                 bins=20, color='red',
                 line_kws={'linewidth': 2},
                 )
    plt.xlim([0.0, 1.0])
    if save:
        file_path = "data/bdd100k" if save_path is None else save_path
        plt.savefig(os.path.join(file_path, "bdd_divergence.pdf"))
    if show:
        plt.show()


def get_label_distribution(data_y, data_index_map):
    """ get the data attribute distributions of each client
    Args:
        data_y (array[int]): A array of data.
        data_index_map (dict): The dict of data index assigned to each client.

    Returns:
        dict[float]: A dict of per-client data distribution.
    """
    data_distribution = {}
    label_num = len(np.unique(data_y))
    for cid, data_idx in enumerate(data_index_map):
        distribution = [0 for _ in range(label_num)]
        unique_values, counts = np.unique(data_y[data_idx], return_counts=True)
        for i in range(len(unique_values)):
            distribution[unique_values[i]] = counts[i]
        data_distribution[cid] = np.array(distribution) / len(data_idx)
    return data_distribution


def get_pairwise_js_divergence(distribution_dict):
    num_client = len(distribution_dict)
    jsd_list = []
    for i in range(num_client):
        for j in range(i + 1, num_client):
            jsd = jensenshannon(distribution_dict[i], distribution_dict[j]) ** 2
            jsd_list.append(jsd)

    return np.array(jsd_list)


if __name__ == '__main__':
    # change to the actual root_path
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_path = os.path.join(root_path, "benchmark/data/bdd100k/bdd100k_iid_100_10_1_0.05_0.1_sample_0.9")
    federated_idx, image_type = np.load(os.path.join(root_path, "federated_idx_attributes.pkl"), allow_pickle=True)
    # display the data distribution
    data_distribution_display(image_type, federated_idx, show=True, save=False)
    pairwise_divergence(image_type, federated_idx, show=True, save=False)
    datasize_distribution(federated_idx, show=True, save=False)
