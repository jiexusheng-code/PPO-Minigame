import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class TBDualWriterCallback(BaseCallback):
    """Write TensorBoard scalars into two separate subdirectories:
    - env-step: x axis = environment steps (`model.num_timesteps`)
    - optimization-step: x axis = optimization/update step counter (opt_step)

    The callback reads `policy._last_per_arg_stats` (if available) to write per-arg metrics.
    Supports write-frequency control and attempts to ensure only a main/root process writes (checks common env vars).
    """

    def __init__(self, base_log_dir: str, write_env_every: int = 1, write_opt_every: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.base_log_dir = base_log_dir
        self.write_env_every = max(1, int(write_env_every))
        self.write_opt_every = max(1, int(write_opt_every))
        self.env_writer: Optional[SummaryWriter] = None
        self.opt_writer: Optional[SummaryWriter] = None
        self.opt_step = 0

    def _on_training_start(self) -> None:
        env_dir = os.path.join(self.base_log_dir, "env-step")
        opt_dir = os.path.join(self.base_log_dir, "optimization-step")
        os.makedirs(env_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        self.env_writer = SummaryWriter(env_dir)
        self.opt_writer = SummaryWriter(opt_dir)

    def _is_main_process(self) -> bool:
        """Try to detect main process using common environment variables. Defaults to True."""
        try:
            rank_vars = ["RANK", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK", "SLURM_PROCID"]
            for v in rank_vars:
                val = os.environ.get(v)
                if val is not None:
                    try:
                        return int(val) == 0
                    except Exception:
                        continue
        except Exception:
            pass
        return True

    def _on_rollout_end(self) -> None:
        # write env-step simple scalar: current num_timesteps
        # env-step writes: only if main process and respecting frequency
        try:
            if not self._is_main_process():
                return
            num_ts = int(self.model.num_timesteps)
            if (num_ts % self.write_env_every) == 0 and self.env_writer is not None:
                self.env_writer.add_scalar("training/num_timesteps", num_ts, num_ts)
        except Exception:
            pass

        # write optimization-step per-arg stats (if provided by policy)
        try:
            policy = getattr(self.model, "policy", None)
            stats = getattr(policy, "_last_per_arg_stats", None)
            if stats is not None and self.opt_writer is not None and self._is_main_process():
                # only write optimization stats every write_opt_every updates
                if (self.opt_step % self.write_opt_every) == 0:
                    step = int(self.opt_step)
                    # fn-level
                    if "fn_entropy_mean" in stats:
                        self.opt_writer.add_scalar("fn/entropy_mean", float(stats["fn_entropy_mean"]), step)
                    # per-slot
                    slot_used = stats.get("slot_used_mean", []) or []
                    slot_ent = stats.get("slot_entropy_mean", []) or []
                    for i, used in enumerate(slot_used):
                        self.opt_writer.add_scalar(f"arg/slot_{i:02d}_used_rate", float(used), step)
                    for i, ent in enumerate(slot_ent):
                        self.opt_writer.add_scalar(f"arg/slot_{i:02d}_entropy_mean", float(ent), step)

        except Exception:
            pass

        # increment opt counter (treat each rollout-end as an optimization step)
        self.opt_step += 1

    def _on_training_end(self) -> None:
        try:
            if self.env_writer is not None:
                self.env_writer.close()
            if self.opt_writer is not None:
                self.opt_writer.close()
        except Exception:
            pass
