import datetime
import gzip
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


# Trajectory sampling is deliberately limited to the first environment and to
# CollectMineralShards (the wrapper only emits the payload for that map).
TRAJECTORY_INFO_KEY = "collect_mineral_shards/trajectory"
TRAJECTORY_ENV_INDEX = 0
TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS = 50_000


class TBDualWriterCallback(BaseCallback):
    """Write project-specific TensorBoard scalars under the ``extra`` directory.

    SB3 owns the default scalar writer under the configured log directory. This
    callback reads `policy._last_per_arg_stats` (if available) to write per-arg
    metrics and uses the optimization step as the x-axis for update metrics.
    Episode-level reward and length remain on an episode counter.
    Supports write-frequency control and attempts to ensure only a main/root
    process writes.
    """

    def __init__(self, base_log_dir: str, write_env_every: int = 1, write_opt_every: int = 1, save_every: int = 0, save_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.base_log_dir = base_log_dir
        self.write_env_every = max(1, int(write_env_every))
        self.write_opt_every = max(1, int(write_opt_every))
        self.save_every = int(save_every) if save_every is not None else 0
        self.save_dir = save_dir
        # only keep the extra writer (per-arg and episode scalars)
        self.opt_writer: Optional[SummaryWriter] = None
        self.opt_step = 0
        self._episode_counter = 0
        self._last_n_updates = None
        # de-dup guard: ensure (writer, tag, step) is written at most once
        self._written_scalar_keys = set()
        self._collect_diag_sum = {}
        self._collect_diag_count = {}
        self._collect_diag_steps = 0
        self._collect_fn_counts = {}
        self._collect_reason_counts = {}
        # CollectMineralShards trajectory sampling state (env 0 only). Only a
        # single streaming file handle and a few counters are retained.
        self._traj_dir = os.path.join(base_log_dir, "trajectories")
        self._traj_file = None
        self._traj_path = None
        self._traj_episode_id = None
        self._traj_start_global = None
        self._traj_steps = 0
        self._traj_reward_sum = 0.0
        self._traj_next_allowed_step = 0

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

    @staticmethod
    def _json_safe(value):
        """Recursively convert numpy scalars and non-finite floats for strict JSON."""
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {
                str(key): TBDualWriterCallback._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TBDualWriterCallback._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return TBDualWriterCallback._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return TBDualWriterCallback._json_safe(value.item())
        return value

    def _write_trajectory_line(self, obj) -> None:
        if self._traj_file is None:
            return
        line = json.dumps(
            self._json_safe(obj),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._traj_file.write(line.encode("utf-8") + b"\n")

    def _open_trajectory(self, payload, global_step: int) -> None:
        os.makedirs(self._traj_dir, exist_ok=True)
        episode_id = int(payload.get("episode_id", -1))
        base = (
            f"collect_mineral_shards_env0_ep{episode_id:06d}"
            f"_g{int(global_step)}_{time.time_ns()}"
        )
        path = os.path.join(self._traj_dir, base + ".jsonl.gz")
        suffix = 1
        # Never overwrite an existing run's file (e.g. after a resume).
        while os.path.exists(path):
            path = os.path.join(self._traj_dir, f"{base}_{suffix}.jsonl.gz")
            suffix += 1
        self._traj_file = gzip.open(path, "wb")
        self._traj_path = path
        self._traj_episode_id = episode_id
        self._traj_start_global = int(global_step)
        self._traj_steps = 0
        self._traj_reward_sum = 0.0
        self._write_trajectory_line({
            "type": "header",
            "schema_version": 1,
            "map": payload.get("map"),
            "env_index": TRAJECTORY_ENV_INDEX,
            "episode_id": episode_id,
            "start_global_num_timesteps": int(global_step),
            "seed": payload.get("seed"),
            "seed_source": payload.get("seed_source", "unknown"),
            "sampling_rule": (
                "first_complete_episode_then_every_"
                f"{int(TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS)}_global_env_steps"
            ),
            "queued_note": (
                "queued_flag is a FunctionCall flag, not a real order-queue length"
            ),
            "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    def _close_trajectory(
        self,
        complete: bool,
        reason: Optional[str] = None,
        global_step: Optional[int] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        if self._traj_file is None:
            return
        start = self._traj_start_global
        path = self._traj_path
        try:
            end = {
                "type": "episode_end",
                "complete": bool(complete),
                "episode_id": self._traj_episode_id,
                "steps": int(self._traj_steps),
                "total_reward": float(self._traj_reward_sum),
                "end_global_num_timesteps": (
                    None if global_step is None else int(global_step)
                ),
            }
            if complete:
                end["terminated"] = None if terminated is None else bool(terminated)
                end["truncated"] = None if truncated is None else bool(truncated)
            else:
                end["incomplete_reason"] = reason or "unknown"
            self._write_trajectory_line(end)
        except Exception:
            # A close-path write failure must stay visible: trajectory data is
            # lost, but PPO training must not be interrupted.
            logging.getLogger(__name__).exception(
                "CollectMineralShards 轨迹结束标记写入失败，路径: %s", path
            )
        finally:
            stream = self._traj_file
            try:
                stream.flush()
            except Exception:
                logging.getLogger(__name__).exception(
                    "CollectMineralShards 轨迹 flush 失败，路径: %s", path
                )
            try:
                stream.close()
            except Exception:
                logging.getLogger(__name__).exception(
                    "CollectMineralShards 轨迹 close 失败，路径: %s", path
                )
            self._traj_file = None
            self._traj_path = None
            self._traj_episode_id = None
            self._traj_start_global = None
            self._traj_steps = 0
            self._traj_reward_sum = 0.0
            if start is not None:
                self._traj_next_allowed_step = (
                    int(start) + int(TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS)
                )

    def _update_trajectory(self, infos) -> None:
        """Stream at most one CollectMineralShards episode from env 0.

        Sampling starts at a real episode boundary (``episode_step == 0``),
        never by slicing into an episode already in progress. After a sampled
        episode starts, the next sample is only allowed once the global env
        step counter has advanced by the configured interval.
        """
        if not isinstance(infos, (list, tuple)) or len(infos) <= TRAJECTORY_ENV_INDEX:
            return
        info = infos[TRAJECTORY_ENV_INDEX]
        if not isinstance(info, dict):
            return
        payload = info.get(TRAJECTORY_INFO_KEY)
        if not isinstance(payload, dict) or payload.get("episode_step") is None:
            return
        if payload.get("map") != "CollectMineralShards":
            return
        global_step = int(getattr(self.model, "num_timesteps", 0))
        episode_step = int(payload["episode_step"])
        episode_id = payload.get("episode_id")
        if self._traj_file is not None and episode_id != self._traj_episode_id:
            # A new episode began without seeing the previous terminal step.
            # Keep the partial stream, but mark it incomplete rather than
            # presenting it as a complete episode.
            self._close_trajectory(
                complete=False,
                reason="episode_boundary_lost",
                global_step=global_step,
            )
        if self._traj_file is None:
            if episode_step != 0 or global_step < self._traj_next_allowed_step:
                return
            self._open_trajectory(payload, global_step)
        self._write_trajectory_line({
            **payload,
            "type": "step",
            "global_num_timesteps": global_step,
        })
        self._traj_steps += 1
        try:
            self._traj_reward_sum += float(payload.get("reward") or 0.0)
        except (TypeError, ValueError):
            pass
        if bool(payload.get("terminated")) or bool(payload.get("done")):
            self._close_trajectory(
                complete=True,
                global_step=global_step,
                terminated=bool(payload.get("terminated")),
                truncated=bool(payload.get("truncated")),
            )

    def close_incomplete_trajectory(self, reason: str = "training_interrupted") -> None:
        """Close an open trajectory stream as incomplete.

        SB3 does not call ``on_training_end`` when ``learn()`` is interrupted
        (e.g. KeyboardInterrupt), so the launcher calls this from its cleanup
        path. On normal completion the stream is already closed and this is a
        no-op.
        """
        try:
            self._close_trajectory(complete=False, reason=reason)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        opt_dir = os.path.join(self.base_log_dir, "extra")
        os.makedirs(opt_dir, exist_ok=True)
        self.opt_writer = SummaryWriter(opt_dir)

    def _on_step(self) -> bool:
        # Required by BaseCallback; no per-step action needed for this writer
        # Capture per-environment infos (Monitor) to log episode-level scalars with episode x-axis.
        infos = None
        try:
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
                    if "collect_mineral_shards/requested_fn_id" in info:
                        self._collect_diag_steps += 1
                        for stage in ("requested", "sanitized", "executed"):
                            fid = info.get(f"collect_mineral_shards/{stage}_fn_id", -1)
                            name = info.get(f"collect_mineral_shards/{stage}_fn_name", "invalid")
                            tag = f"collect_mineral_shards/{stage}_function_fraction/{fid}_{name}"
                            self._collect_fn_counts[tag] = self._collect_fn_counts.get(tag, 0) + 1
                        reason = info.get("collect_mineral_shards/fallback_reason_code", "none")
                        if isinstance(reason, str):
                            self._collect_reason_counts[reason] = self._collect_reason_counts.get(reason, 0) + 1
                    for key, value in info.items():
                        if not str(key).startswith("collect_mineral_shards/") or isinstance(value, (str, tuple, list, dict)):
                            continue
                        # Categorical identifiers and flattened spatial indices
                        # are not meaningful numeric averages; function
                        # distributions are tracked as *_function_fraction tags
                        # and spatial targets as decoded x/y scalars.
                        if str(key).endswith(("_fn_id", "_flat")):
                            continue
                        try:
                            scalar = float(value)
                            if not math.isfinite(scalar):
                                continue
                            self._collect_diag_sum[key] = self._collect_diag_sum.get(key, 0.0) + scalar
                            self._collect_diag_count[key] = self._collect_diag_count.get(key, 0) + 1
                        except (TypeError, ValueError):
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
        try:
            self._update_trajectory(infos)
        except Exception:
            logging.getLogger(__name__).exception(
                "CollectMineralShards 轨迹写入失败，关闭当前文件并标记 incomplete"
            )
            try:
                self._close_trajectory(complete=False, reason="callback_error")
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
        if self.opt_writer is not None and self._is_main_process():
            step = int(getattr(self.model, "num_timesteps", 0))
            for key, total in self._collect_diag_sum.items():
                count = self._collect_diag_count.get(key, 0)
                if count:
                    self._safe_add_scalar("opt", self.opt_writer, f"{key}_mean", total / count, step)
            if self._collect_diag_steps:
                for tag, count in self._collect_fn_counts.items():
                    self._safe_add_scalar("opt", self.opt_writer, tag, count / self._collect_diag_steps, step)
                for reason, count in self._collect_reason_counts.items():
                    tag = f"collect_mineral_shards/fallback_reason_fraction/{reason}"
                    self._safe_add_scalar("opt", self.opt_writer, tag, count / self._collect_diag_steps, step)
        self._collect_diag_sum.clear()
        self._collect_diag_count.clear()
        self._collect_fn_counts.clear()
        self._collect_reason_counts.clear()
        self._collect_diag_steps = 0
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
        """Retained as a compatibility hook; SB3 scalars are not mirrored here."""
        # function intentionally removed: this callback no longer mirrors SB3 event files
        return

    def _on_training_end(self) -> None:
        try:
            self._close_trajectory(complete=False, reason="training_end")
        except Exception:
            pass
        try:
            if self.opt_writer is not None:
                self.opt_writer.close()
        except Exception:
            pass
