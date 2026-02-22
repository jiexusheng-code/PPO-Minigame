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
        self._last_n_updates = None
        # track SB3 event files we have mirrored already to avoid duplicate processing
        self._processed_sb3_files = set()
        # de-dup guard: ensure (writer, tag, step) is written at most once
        self._written_scalar_keys = set()

    def _safe_add_scalar(self, writer_name: str, writer: Optional[SummaryWriter], tag: str, value: float, step: int):
        try:
            if writer is None:
                return
            k = (str(writer_name), str(tag), int(step))
            if k in self._written_scalar_keys:
                return
            writer.add_scalar(tag, float(value), int(step))
            self._written_scalar_keys.add(k)
        except Exception:
            pass

    def _dump_logger_scalars(self, logger, writer, step: int, writer_name: str = "env", include_prefixes=None):
        """Write numeric scalars from SB3 logger into provided SummaryWriter."""
        try:
            if logger is None or writer is None:
                return
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
            for k, v in stats.items():
                try:
                    if include_prefixes is not None:
                        if not any(str(k).startswith(p) for p in include_prefixes):
                            continue
                    if v is None:
                        continue
                    # handle common wrapper types: tuple (value, fmt), objects with .value/.mean, numpy types
                    val = None
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        candidate = v[0]
                    else:
                        candidate = v
                    try:
                        # numpy scalar
                        import numbers
                        if isinstance(candidate, numbers.Number):
                            val = float(candidate)
                        else:
                            # objects like (value, fmt) or small wrappers
                            if hasattr(candidate, 'value'):
                                val = float(getattr(candidate, 'value'))
                            elif hasattr(candidate, 'mean'):
                                val = float(getattr(candidate, 'mean'))
                            else:
                                val = float(candidate)
                    except Exception:
                        # final fallback: try converting first element if iterable
                        try:
                            val = float(list(candidate)[0])
                        except Exception:
                            continue
                    self._safe_add_scalar(writer_name, writer, k, val, int(step))
                except Exception:
                    continue
        except Exception:
            pass

    def _on_training_start(self) -> None:
        env_dir = os.path.join(self.base_log_dir, "fundamental")
        opt_dir = os.path.join(self.base_log_dir, "extra")
        os.makedirs(env_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        self.env_writer = SummaryWriter(env_dir)
        self.opt_writer = SummaryWriter(opt_dir)
        # Add a custom SB3-compatible output format that writes SB3 logger
        # key/value pairs into our `fundamental` SummaryWriter but using
        # optimization/update steps (`opt_step`) as the x-axis. This ensures
        # SB3's `train/*`, `rollout/*`, `eval/*` scalars appear in `fundamental`
        # with the correct step semantics.
        try:
            class _OptStepTBOutputFormat:
                def __init__(self, writer, get_step_fn, add_scalar_fn):
                    self.writer = writer
                    self.get_step = get_step_fn
                    self.add_scalar = add_scalar_fn

                def writekvs(self, kvs):
                    try:
                        step = int(self.get_step())
                    except Exception:
                        step = 0
                    try:
                        import numbers
                        for k, v in (kvs or {}).items():
                            try:
                                if v is None:
                                    continue
                                # handle common wrappers
                                val = None
                                if isinstance(v, numbers.Number):
                                    val = float(v)
                                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], numbers.Number):
                                    val = float(v[0])
                                elif hasattr(v, 'value'):
                                    val = float(getattr(v, 'value'))
                                elif hasattr(v, 'mean'):
                                    val = float(getattr(v, 'mean'))
                                else:
                                    try:
                                        val = float(v)
                                    except Exception:
                                        continue
                                try:
                                    self.add_scalar(k, val, step)
                                except Exception:
                                    continue
                            except Exception:
                                continue
                    except Exception:
                        pass

                def write(self, *args, **kwargs):
                    pass

                def flush(self):
                    try:
                        self.writer.flush()
                    except Exception:
                        pass

                def close(self):
                    try:
                        self.writer.flush()
                    except Exception:
                        pass

            model_logger = getattr(getattr(self, 'model', None), 'logger', None)
            if model_logger is not None:
                try:
                    # Append our custom format if not already present
                    existing = getattr(model_logger, 'output_formats', [])
                    try:
                        existing_list = list(existing)
                    except Exception:
                        existing_list = []
                    already = False
                    for fmt in existing_list:
                        if getattr(fmt, '__class__', None).__name__ == '_OptStepTBOutputFormat':
                            already = True
                            break
                    if not already:
                        existing_list.append(
                            _OptStepTBOutputFormat(
                                self.env_writer,
                                lambda: self.opt_step,
                                lambda tag, value, step: self._safe_add_scalar("env", self.env_writer, tag, value, step),
                            )
                        )
                        try:
                            model_logger.output_formats = existing_list
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

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
                                self._safe_add_scalar('opt', self.opt_writer, 'sc2/episode_score', float(score), step)
                        except Exception:
                            pass
                        try:
                            if length is not None and self.opt_writer is not None:
                                self._safe_add_scalar('opt', self.opt_writer, 'sc2/episode_length', float(length), step)
                        except Exception:
                            pass
        except Exception:
            pass

        # write train/* metrics once per optimizer update
        try:
            if self._is_main_process():
                logger = getattr(getattr(self, 'model', None), 'logger', None)
                n_updates = self._get_logger_n_updates(logger)
                if n_updates is not None and n_updates != self._last_n_updates:
                    self._dump_logger_scalars(
                        logger,
                        self.env_writer,
                        int(self.opt_step),
                        writer_name="env",
                        include_prefixes=("train/",),
                    )
                    self._last_n_updates = n_updates
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

    def _get_logger_n_updates(self, logger):
        try:
            if logger is None:
                return None
            name_to_value = getattr(logger, 'name_to_value', None)
            if isinstance(name_to_value, dict):
                if 'train/n_updates' in name_to_value:
                    return int(name_to_value.get('train/n_updates'))
                if 'n_updates' in name_to_value:
                    return int(name_to_value.get('n_updates'))
            name_to_mean = getattr(logger, 'name_to_mean', None)
            if isinstance(name_to_mean, dict):
                if 'train/n_updates' in name_to_mean:
                    return int(name_to_mean.get('train/n_updates'))
                if 'n_updates' in name_to_mean:
                    return int(name_to_mean.get('n_updates'))
        except Exception:
            return None
        return None

    def _on_rollout_end(self) -> None:
        # write env-step simple scalar: current num_timesteps
        # env-step writes: only if main process and respecting frequency
        try:
            if not self._is_main_process():
                return
            num_ts = int(self.model.num_timesteps)
            # write using optimization-step as x-axis (treat rollout count as primary step)
            if (self.opt_step % self.write_env_every) == 0 and self.env_writer is not None:
                self._safe_add_scalar("env", self.env_writer, "training/num_timesteps", num_ts, int(self.opt_step))
                # write non-train scalar groups on rollout boundary
                try:
                    logger = getattr(self.model, "logger", None)
                    self._dump_logger_scalars(
                        logger,
                        self.env_writer,
                        int(self.opt_step),
                        writer_name="env",
                        include_prefixes=("eval/", "rollout/", "time/"),
                    )
                except Exception:
                    pass
                # robust eval metric export (independent of logger internals)
                try:
                    eval_stats = getattr(self.model, "_last_eval_stats", None)
                    if isinstance(eval_stats, dict):
                        if "mean_reward" in eval_stats:
                            self._safe_add_scalar("env", self.env_writer, "eval/mean_reward", float(eval_stats["mean_reward"]), int(self.opt_step))
                        if "mean_ep_length" in eval_stats:
                            self._safe_add_scalar("env", self.env_writer, "eval/mean_ep_length", float(eval_stats["mean_ep_length"]), int(self.opt_step))
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
                            self._safe_add_scalar("opt", self.opt_writer, "entropy/fn", float(stats["fn_entropy_mean"]), step)
                        except Exception:
                            pass
                    if "fn_logprob_mean" in stats:
                        try:
                            self._safe_add_scalar("opt", self.opt_writer, "log_prob/fn", float(stats["fn_logprob_mean"]), step)
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
                            self._safe_add_scalar("opt", self.opt_writer, f"used/arg/{name}", float(used), step)
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
                            self._safe_add_scalar("opt", self.opt_writer, f"entropy/arg/{name}", float(ent), step)
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
                            self._safe_add_scalar("opt", self.opt_writer, f"log_prob/arg/{name}", float(lp), step)
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

    def _mirror_sb3_tb_to_fundamental(self) -> None:
        """Scan for SB3 TensorBoard event files under the base log dir (excluding our own
        fundamental/extra dirs) and mirror any new scalar values into the `fundamental`
        SummaryWriter using the current `opt_step` as x-axis. Files already processed are
        tracked in `self._processed_sb3_files` to avoid duplicate writes.
        """
        try:
            if self.env_writer is None:
                return
            import glob
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            except Exception:
                return

            # look for event files recursively under base_log_dir but skip our own subdirs
            pattern = os.path.join(self.base_log_dir, "**", "events.out.tfevents.*")
            for path in glob.glob(pattern, recursive=True):
                # skip our own writers' dirs
                if os.path.commonpath([os.path.abspath(path), os.path.abspath(os.path.join(self.base_log_dir, 'fundamental'))]) == os.path.abspath(os.path.join(self.base_log_dir, 'fundamental')):
                    continue
                if os.path.commonpath([os.path.abspath(path), os.path.abspath(os.path.join(self.base_log_dir, 'extra'))]) == os.path.abspath(os.path.join(self.base_log_dir, 'extra')):
                    continue
                if path in self._processed_sb3_files:
                    continue
                try:
                    ea = EventAccumulator(path, size_guidance={
                        EventAccumulator.SCALARS: 0,
                    })
                    ea.Reload()
                except Exception:
                    # skip unreadable files
                    self._processed_sb3_files.add(path)
                    continue
                try:
                    tags = ea.Tags().get('scalars', [])
                except Exception:
                    tags = []
                for tag in tags:
                    try:
                        events = ea.Scalars(tag)
                        if not events:
                            continue
                        # take the last scalar value recorded in that file
                        ev = events[-1]
                        val = float(ev.value)
                        # write into fundamental using current opt_step
                        try:
                            self.env_writer.add_scalar(tag, val, int(self.opt_step))
                        except Exception:
                            continue
                    except Exception:
                        continue
                # mark file processed
                self._processed_sb3_files.add(path)
        except Exception:
            pass

    def _on_training_end(self) -> None:
        try:
            if self.env_writer is not None:
                self.env_writer.close()
            if self.opt_writer is not None:
                self.opt_writer.close()
        except Exception:
            pass
