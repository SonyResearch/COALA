import coala

config = {
    "data": {"dataset": "cifar10"},
    "model": "simple_cnn",
    "test_mode": "test_in_server"
}
coala.init(config)
coala.run()
