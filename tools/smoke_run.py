import copy
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Make the repository root importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train


def build_smoke_config(base_cfg):
    """Return a shallow-unshared smoke-run copy of base_cfg."""
    cfg = copy.deepcopy(base_cfg)
    cfg["n_envs"] = 1
    cfg["total_timesteps"] = 512
    # eval_every=0 -> eval_freq=0. SB3's EvalCallback skips evaluation when
    # eval_freq <= 0, and train.py's custom initial _on_step() call goes through
    # that same guard, so no eval episode is run during the smoke test.
    cfg["eval_every"] = 0
    cfg.setdefault("env_kwargs", {})
    cfg["env_kwargs"]["visualize"] = False

    # A rollout of n_envs * n_steps is smaller than the production batch_size in
    # this smoke setup; clamp it so SB3 does not truncate minibatches.
    rollout_size = int(cfg["n_envs"]) * int(cfg["n_steps"])
    cfg["batch_size"] = min(int(cfg["batch_size"]), rollout_size)
    return cfg


def main():
    cfg = build_smoke_config(train.load_config(train.DEFAULT_CONFIG_PATH))

    # Keep smoke-run models/logs out of the production out_dir.
    smoke_out_dir = tempfile.mkdtemp(prefix="ppo_smoke_")
    cfg["out_dir"] = smoke_out_dir

    def fake_load_config(path):
        return cfg

    train.load_config = fake_load_config

    print("Starting smoke run with modified config:")
    print(cfg)
    try:
        train.main()
    finally:
        # train.main() attaches a FileHandler to this temp dir; close it first
        # so Windows can delete the directory.
        logging.shutdown()
        for attempt in range(5):
            try:
                shutil.rmtree(smoke_out_dir)
                break
            except FileNotFoundError:
                break
            except Exception as exc:
                if attempt == 4:
                    print(f"WARNING: could not remove smoke output dir {smoke_out_dir}: {exc}")
                else:
                    time.sleep(0.2)


if __name__ == "__main__":
    main()
