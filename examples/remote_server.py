import coala

config = {
    "data": {"dataset": "femnist"},
    "model": "lenet",
    "test_mode": "test_in_client",
    "server": {"rounds": 5, "clients_per_round": 2},
    "client": {"track": True},
}

coala.start_remote_server(config)
