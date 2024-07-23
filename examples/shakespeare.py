import coala

config = {
    "data": {"dataset": "shakespeare"},
    "model": "rnn",
    "test_mode": "test_in_client"
}
coala.init(config)
coala.run()
