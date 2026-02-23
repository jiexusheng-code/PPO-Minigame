import os
import train

cfg = train.load_config(train.DEFAULT_CONFIG_PATH)
cfg["n_envs"] = 1
cfg["total_timesteps"] = 512
cfg["eval_every"] = 1000000  # avoid running eval during smoke test (optimization-step units)
cfg["env_kwargs"] = cfg.get("env_kwargs", {})
cfg["env_kwargs"]["visualize"] = False

def fake_load_config(path):
    return cfg

train.load_config = fake_load_config

if __name__ == "__main__":
    print("Starting smoke run with modified config:")
    print(cfg)
    train.main()
