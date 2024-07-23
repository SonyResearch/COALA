import coala

from coala.distributed import slurm

rank, local_rank, world_size, host_addr = slurm.setup()

configs = {
    "gpu": 4,
    "distributed": {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "init_method": host_addr,
    },
    "server": {
        "clients_per_round": 20,
    }
}

coala.init(configs)
coala.run()
