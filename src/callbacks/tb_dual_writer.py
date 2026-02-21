import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class TBDualWriterCallback(BaseCallback):
    """Write TensorBoard scalars into two separate subdirectories:
    - fundamental: x axis = optimization/update step counter (opt_step)
    - extra: x axis = optimization/update step counter (opt_step)

    NOTE: All scalars (except the episode-level reward plot which remains on episode x-axis)
    are written with the optimization-step as the x-axis. The callback reads
    `policy._last_per_arg_stats` (if available) to write per-arg metrics.
    Supports write-frequency control and attempts to ensure only a main/root process writes.
    """

    def __init__(self, base_log_dir: str, write_env_every: int = 1, write_opt_every: int = 1, save_every: int = 0, save_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.base_log_dir = base_log_dir
        self.write_env_every = max(1, int(write_env_every))
        self.write_opt_every = max(1, int(write_opt_every))
        self.save_every = int(save_every) if save_every is not None else 0
        self.save_dir = save_dir
        self.env_writer: Optional[SummaryWriter] = None
        self.opt_writer: Optional[SummaryWriter] = None
        self.opt_step = 0
        self._episode_counter = 0

    def _on_training_start(self) -> None:
        env_dir = os.path.join(self.base_log_dir, "fundamental")
        opt_dir = os.path.join(self.base_log_dir, "extra")
        os.makedirs(env_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        self.env_writer = SummaryWriter(env_dir)
        self.opt_writer = SummaryWriter(opt_dir)

    def _on_step(self) -> bool:
        # Required by BaseCallback; no per-step action needed for this writer
        # Capture per-environment infos (Monitor) to log episode-level scalars with episode x-axis.
        try:
            infos = None
            if hasattr(self, "locals"):
                infos = self.locals.get("infos") if isinstance(self.locals, dict) else None
            if infos is None:
                # try model attribute (vec_env wrapper may expose last infos)
                try:
                    infos = getattr(self.model, 'env', None)
                except Exception:
                    infos = None
            if infos and isinstance(infos, (list, tuple)):
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    ep = info.get("episode") or info.get("episode_info") or None
                    if ep is not None:
                        # episode dict commonly contains 'r' (reward) and 'l' (length)
                        try:
                            score = ep.get('r') if isinstance(ep, dict) else None
                            length = ep.get('l') if isinstance(ep, dict) else None
                        except Exception:
                            score = None
                            length = None
                        # increment episode counter and write to extra with episode index as x-axis
                        self._episode_counter += 1
                        step = int(self._episode_counter)
                        try:
                            if score is not None and self.opt_writer is not None:
                                self.opt_writer.add_scalar('sc2/episode_score', float(score), step)
                        except Exception:
                            pass
                        try:
                            if length is not None and self.opt_writer is not None:
                                self.opt_writer.add_scalar('sc2/episode_length', float(length), step)
                        except Exception:
                            pass
        except Exception:
            pass
        return True

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
            # write using optimization-step as x-axis (treat rollout count as primary step)
            if (self.opt_step % self.write_env_every) == 0 and self.env_writer is not None:
                self.env_writer.add_scalar("training/num_timesteps", num_ts, int(self.opt_step))
                # Also attempt to capture SB3's logger scalar values (train/*, rollout/*)
                try:
                    logger = getattr(self.model, "logger", None)
                    if logger is not None and self.env_writer is not None:
                        # SB3 Logger exposes name_to_value and name_to_mean dicts
                        stats = {}
                        if hasattr(logger, "name_to_value"):
                            try:
                                stats.update({k: v for k, v in logger.name_to_value.items()})
                            except Exception:
                                pass
                        if hasattr(logger, "name_to_mean"):
                            try:
                                stats.update({k: v for k, v in logger.name_to_mean.items()})
                            except Exception:
                                pass
                        # Write selected scalars into fundamental timeline using opt_step as x-axis
                        for k, v in stats.items():
                            try:
                                if v is None:
                                    continue
                                # ensure numeric
                                val = float(v)
                                # prefix cleanup: logger keys often contain '/' which is fine for TB tags
                                self.env_writer.add_scalar(k, val, int(self.opt_step))
                            except Exception:
                                continue
                except Exception:
                    pass
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
                        try:
                            self.opt_writer.add_scalar("entropy/fn", float(stats["fn_entropy_mean"]), step)
                        except Exception:
                            pass
                    if "fn_logprob_mean" in stats:
                        try:
                            self.opt_writer.add_scalar("log_prob/fn", float(stats["fn_logprob_mean"]), step)
                        except Exception:
                            pass
                    # per-slot
                    slot_used = stats.get("slot_used_mean", []) or []
                    slot_ent = stats.get("slot_entropy_mean", []) or []
                    slot_logp = stats.get("slot_logprob_mean", []) or []
                    # try to map slot index -> semantic name from policy (if available)
                    param_names = None
                    try:
                        param_names = getattr(policy, "_param_semantics", None)
                    except Exception:
                        param_names = None

                    for i, used in enumerate(slot_used):
                        name = None
                        try:
                            if param_names and i < len(param_names):
                                name = str(param_names[i])
                            else:
                                name = f"slot_{i:02d}"
                        except Exception:
                            name = f"slot_{i:02d}"
                        try:
                            self.opt_writer.add_scalar(f"used/arg/{name}", float(used), step)
                        except Exception:
                            pass

                    for i, ent in enumerate(slot_ent):
                        name = None
                        try:
                            if param_names and i < len(param_names):
                                name = str(param_names[i])
                            else:
                                name = f"slot_{i:02d}"
                        except Exception:
                            name = f"slot_{i:02d}"
                        try:
                            self.opt_writer.add_scalar(f"entropy/arg/{name}", float(ent), step)
                        except Exception:
                            pass

                    for i, lp in enumerate(slot_logp):
                        name = None
                        try:
                            if param_names and i < len(param_names):
                                name = str(param_names[i])
                            else:
                                name = f"slot_{i:02d}"
                        except Exception:
                            name = f"slot_{i:02d}"
                        try:
                            self.opt_writer.add_scalar(f"log_prob/arg/{name}", float(lp), step)
                        except Exception:
                            pass

            # periodic model checkpointing by rollout/update counts
            if self.save_every and self._is_main_process():
                try:
                    if (self.opt_step % self.save_every) == 0:
                        if self.save_dir is not None:
                            fname = os.path.join(self.save_dir, f"checkpoint_opt_{self.opt_step}")
                        else:
                            fname = os.path.join(self.base_log_dir, f"checkpoint_opt_{self.opt_step}")
                        # SB3 model is available as self.model
                        try:
                            self.model.save(fname)
                        except Exception:
                            pass
                except Exception:
                    pass
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
