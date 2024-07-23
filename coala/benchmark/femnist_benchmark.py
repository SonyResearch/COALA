import coala


def benchmark_iid():
    """
    Benchmark the performance of FEMNIST dataset with 350 clients on iid setting.
    Expected test accuracy: 
        - 54.70% at round 20 
    """
    config = {
        "task_id": "femnist",
        "data": {
            "dataset": "femnist",
            "split_type": "iid",
        },
        "server": {
            "rounds": 20,
            "clients_per_round": 5,
            "test_all": True,
            "test_every": 10
        },
        "client": {"local_epoch": 5},
        "model": "lenet",
        "test_mode": "test_in_client",
    }

    coala.init(config)
    coala.run()


def benchmark_non_iid():
    """
    Benchmark the performance of FEMNIST dataset with 186 clients on niid setting.
    Expected test accuracy: 
        - 65.15% at round 20
    """
    config = {
        "task_id": "femnist",
        "data": {
            "dataset": "femnist",
            "split_type": "niid"
        },
        "server": {
            "rounds": 20,
            "clients_per_round": 5,
            "test_all": True,
            "test_every": 10,
        },
        "client": {"local_epoch": 5},
        "model": "lenet",
        "test_mode": "test_in_client",
    }

    coala.init(config)
    coala.run()


if __name__ == "__main__":
    benchmark_iid()
    benchmark_non_iid()
