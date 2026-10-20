"""Minimal training launcher using stable-baselines3 PPO, driven by config file."""

import os
import glob
import copy
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor
from src.callbacks.tb_dual_writer import TBDualWriterCallback

DEFAULT_CONFIG_PATH = os.environ.get("TRAIN_CONFIG_PATH", "./configs/ppo_config.yaml")

PPO_PARAM_KEYS = (
    "learning_rate", "ent_coef", "batch_size", "n_epochs", "gamma",
    "gae_lambda", "n_steps", "clip_range", "vf_coef", "max_grad_norm",
)

REQUIRED_CONFIG_KEYS = (
    "env", "out_dir", "n_envs", "seed", "device", "total_timesteps", "policy",
    "policy_kwargs", "env_kwargs",
) + PPO_PARAM_KEYS + (
    "save_iters", "summary_iters", "eval_every", "n_eval_episodes",
    "eval_deterministic", "eval_render", "tensorboard", "tb_log_dir",
    "save_best_model", "verbose", "log_interval",
)

# Evaluation environments use a disjoint seed range so their episodes never
# replay the training environments' initial states.
EVAL_SEED_OFFSET = 100


def compute_env_seed(base_seed, rank, offset=0):
    """Return the deterministic seed for environment ``rank``.

    Training envs use ``base_seed + rank``; evaluation envs pass an offset and
    therefore use an independent range.
    """
    if base_seed is None:
        raise ValueError("seed must be an integer for reproducible per-env seeding")
    seed = int(base_seed) + int(offset) + int(rank)
    if seed < 0:
        raise ValueError(f"Computed a negative environment seed: {seed}")
    return seed


def compute_eval_seed_offset(n_envs):
    """Start the eval seed range after every training rank."""
    return max(EVAL_SEED_OFFSET, int(n_envs))


def compute_learn_target(requested_timesteps, current_num_timesteps, reset_num_timesteps):
    """Convert a requested number of additional steps into SB3's absolute target.

    With ``reset_num_timesteps=True`` SB3 interprets ``total_timesteps`` from
    zero. With ``reset_num_timesteps=False`` it compares against the restored
    counter, so the target must be offset by the current value to append the
    requested number of steps instead of silently running fewer.
    """
    requested = int(requested_timesteps)
    if reset_num_timesteps:
        return requested
    return int(current_num_timesteps) + requested


def compute_resume_offsets(num_timesteps, n_envs, n_steps):
    """Convert restored environment steps into callback counter offsets."""
    n_envs = max(1, int(n_envs))
    n_steps = max(1, int(n_steps))
    num_timesteps = max(0, int(num_timesteps))
    return {
        "vec_steps": num_timesteps // n_envs,
        "opt_steps": num_timesteps // (n_envs * n_steps),
    }



def resolve_device(device_cfg="auto", *, cuda_available=None, cuda_device_count=None, logger=None):
    """Resolve the configured device to a value understood by SB3/PyTorch.

    - "auto" (default): CUDA when available, otherwise CPU.
    - "cpu": always CPU.
    - "cuda" / "cuda:N": require CUDA; fail fast with a clear error when
      CUDA is unavailable, since silently training on CPU on a GPU host can
      waste a long run. ``cuda:N`` must also reference an existing device.

    ``cuda_available`` and ``cuda_device_count`` are injectable so device
    validation can be tested without a GPU.
    """
    value = "auto" if device_cfg is None else str(device_cfg).strip().lower()
    if not value:
        value = "auto"

    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()

    if value == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif value == "cpu":
        device = "cpu"
    elif value == "cuda" or value.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                f"Config device='{device_cfg}' requires CUDA, but torch.cuda.is_available() "
                "is False. Install a CUDA-enabled PyTorch build and check the NVIDIA "
                "driver, or set device: auto (or cpu)."
            )
        if value.startswith("cuda:"):
            index_text = value.split(":", 1)[1]
            if not index_text.isdigit():
                raise ValueError(
                    f"Invalid CUDA device index in '{device_cfg}'; expected 'cuda:<non-negative integer>'."
                )
            index = int(index_text)
            if cuda_device_count is None:
                import torch
                cuda_device_count = torch.cuda.device_count()
            if index >= int(cuda_device_count):
                raise ValueError(
                    f"CUDA device index {index} in '{device_cfg}' is out of range: "
                    f"torch.cuda.device_count()={int(cuda_device_count)}. "
                    f"Use 'cuda:0'..'cuda:{int(cuda_device_count) - 1}' or 'cuda'."
                )
        device = value
    else:
        raise ValueError(
            f"Unsupported device '{device_cfg}'; expected 'auto', 'cpu', 'cuda', or 'cuda:<index>'."
        )

    if logger is not None:
        logger.info(f"设备配置: device={device_cfg!r} -> {device}")
    return device


def describe_device(device: str) -> str:
    """Human-readable device details for startup logs."""
    import torch
    if device.startswith("cuda"):
        index = torch.cuda.current_device() if device == "cuda" else int(device.split(":", 1)[1])
        return f"PyTorch 设备: {device} ({torch.cuda.get_device_name(index)}), CUDA {torch.version.cuda}"
    return f"PyTorch 设备: cpu (torch {torch.__version__}, CUDA available={torch.cuda.is_available()})"


def make_env_fn(map_name: str, env_kwargs=None, random_seed=None):
    from src.envs.pysc2_gym_wrapper import PySC2GymEnv
    env_kwargs = dict(env_kwargs or {})
    if random_seed is not None:
        # Per-env seed, forwarded to PySC2 at construction time.
        env_kwargs["random_seed"] = int(random_seed)

    def _init():
        return PySC2GymEnv(map_name=map_name, **env_kwargs)

    return _init


def make_seeded_vec_env(map_name: str, env_kwargs, n_envs, seed, offset=0):
    """Build a DummyVecEnv whose env ``rank`` gets ``seed + offset + rank``.

    This mirrors SB3's ``make_vec_env`` (seed + rank) but also passes the same
    deterministic seed to the environment constructor, so PySC2 receives a
    ``random_seed`` before the construction-time probe reset. Resets are seeded
    through the regular ``VecEnv.seed`` path as well.
    """
    n_envs = int(n_envs)
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    env_fns = []
    for rank in range(n_envs):
        env_seed = compute_env_seed(seed, rank, offset)
        factory = make_env_fn(map_name, env_kwargs, random_seed=env_seed)

        def _init(factory=factory, env_seed=env_seed):
            env = factory()
            env.action_space.seed(env_seed)
            return Monitor(env)

        env_fns.append(_init)

    vec_env = DummyVecEnv(env_fns)
    # DummyVecEnv.seed sets _seeds = [seed + i], delivered to reset() once.
    vec_env.seed(compute_env_seed(seed, 0, offset))
    return vec_env


def resolve_policy_and_kwargs(policy_cfg, policy_kwargs):
    """Resolve the runtime policy and return an isolated kwargs copy.

    The on-disk config is never mutated: ``features_extractor_class`` is a
    runtime class object and must not leak into ``config_used.yaml``. A string
    left in the config by an older snapshot is rejected rather than being
    passed to PPO as if it were callable.
    """
    runtime_kwargs = copy.deepcopy(policy_kwargs or {})
    policy = MaskedFlattenPolicy if policy_cfg == "MaskedFlattenPolicy" else policy_cfg
    extractor = runtime_kwargs.get("features_extractor_class")
    if isinstance(extractor, str):
        raise ValueError(
            "policy_kwargs.features_extractor_class must be a class object, not a string "
            f"({extractor!r}). Remove the key to use the project default "
            "VectorLayerNormExtractor; config_used.yaml records runtime classes "
            "under _snapshot only."
        )
    if policy is MaskedFlattenPolicy and extractor is None:
        runtime_kwargs["features_extractor_class"] = VectorLayerNormExtractor
    return policy, runtime_kwargs


def sanitize_for_yaml(obj):
    """Make a config snapshot YAML-safe by turning callables into stable names."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if callable(obj) and not isinstance(obj, (str, bytes)):
        module = getattr(obj, "__module__", "")
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        return f"{module}.{qualname}" if module else str(qualname)
    return obj


def build_config_snapshot(cfg, resolved, source_config):
    """Return a rerunnable config snapshot plus informational runtime metadata.

    ``resolved`` holds runtime-only objects (classes). They are recorded as
    stable strings under ``_snapshot`` and are never fed back to PPO, so the
    config keys themselves remain loadable via ``TRAIN_CONFIG_PATH``.
    """
    snapshot = copy.deepcopy(cfg)
    snapshot["_snapshot"] = {
        "source_config": os.path.abspath(source_config),
        "resolved": sanitize_for_yaml(dict(resolved)),
        "note": (
            "Informational only. Runtime classes are recorded as stable strings "
            "here and are not passed back to PPO. The config keys above are the "
            "rerunnable snapshot."
        ),
    }
    return sanitize_for_yaml(snapshot)

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_eval_freq(eval_every, n_steps):
    """Convert eval_every (optimization-steps) to EvalCallback.eval_freq.

    SB3 calls EvalCallback.on_step once per VecEnv step, where all n_envs
    advance together. Since 1 optimization-step = n_steps VecEnv steps,
    eval_freq must not be multiplied by n_envs.
    """
    return int(eval_every) * int(n_steps)


def build_evaluation(eval_freq, *, logger, make_env, build_callback):
    """Create the eval environment and callback, or ``(None, None)``.

    ``eval_freq <= 0`` disables evaluation entirely: no second SC2 instance is
    spawned and no EvalCallback is built.
    """
    if int(eval_freq) <= 0:
        logger.info("[EvalCallback] 评估已禁用: eval_freq<=0，不创建评估环境/回调。")
        return None, None
    eval_env = make_env()
    try:
        return eval_env, build_callback(eval_env)
    except BaseException:
        # Do not leak the SC2 process if callback construction fails; re-raise
        # the original error.
        try:
            eval_env.close()
        except Exception:
            pass
        raise


def should_prune_periodic_checkpoints(save_best_model, eval_enabled):
    """Periodic checkpoints are only redundant when a best model is saved."""
    return bool(save_best_model) and bool(eval_enabled)


def build_callbacks(*, eval_callback, out_dir, n_steps, n_envs, save_iters,
                    summary_iters, tensorboard, tb_log):
    """Assemble callbacks; checkpointing and TensorBoard never depend on eval."""
    callbacks = []
    if eval_callback is not None:
        callbacks.append(eval_callback)

    save_iters_int = int(save_iters) if save_iters is not None else 0
    if save_iters_int > 0:
        callbacks.append(
            PeriodicCheckpointCallback(out_dir, save_iters_int, n_steps, n_envs)
        )

    tb_callback = None
    if tensorboard and tb_log:
        tb_callback = TBDualWriterCallback(
            tb_log,
            write_env_every=summary_iters,
            write_opt_every=summary_iters,
            save_every=0,
            save_dir=out_dir,
        )
        callbacks.append(tb_callback)
    return callbacks, tb_callback


class PeriodicCheckpointCallback(BaseCallback):
    """Save ``checkpoint_opt_<global opt step>`` every ``save_every`` rollouts.

    This is independent of TensorBoard, so ``save_iters`` is honored even when
    ``tensorboard: false``. The counter is offset by the restored
    ``num_timesteps`` so a resumed run does not re-save an already-passed
    boundary.
    """

    def __init__(self, save_dir, save_every, n_steps, n_envs, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_every = int(save_every)
        self.env_steps_per_opt = max(1, int(n_steps) * int(n_envs))
        self.opt_step = 0

    def _on_step(self) -> bool:
        # Saving is driven by _on_rollout_end; never stop training here.
        return True

    def _on_training_start(self) -> None:
        if self.save_every <= 0:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        self.opt_step = int(getattr(self.model, "num_timesteps", 0)) // self.env_steps_per_opt

    def _on_rollout_end(self) -> None:
        if self.save_every <= 0:
            return
        self.opt_step += 1
        if self.opt_step % self.save_every != 0:
            return
        path = os.path.join(self.save_dir, f"checkpoint_opt_{self.opt_step}")
        try:
            self.model.save(path)
        except Exception as exc:
            logging.getLogger("train").error(f"周期性保存失败: {path}: {exc}")


def main():

    cfg = load_config(DEFAULT_CONFIG_PATH)
    # Strict config access: require keys to be present in the config file.
    def require(key: str):
        if key not in cfg:
            raise KeyError(f"Required config key '{key}' missing from {DEFAULT_CONFIG_PATH}")
        return cfg[key]

    # forbid legacy/ambiguous key 'eval_freq' to avoid silent misconfiguration
    if "eval_freq" in cfg:
        raise ValueError("Config key 'eval_freq' is not supported; use 'eval_every' (optimization-step units) instead.")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in cfg]
    if missing_keys:
        raise KeyError(
            f"Required config key(s) missing from {DEFAULT_CONFIG_PATH}: {', '.join(missing_keys)}"
        )

    env_name = require("env")
    import datetime
    today_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Use user-provided out_dir as base path (required). Timestamped subfolder will be created inside it.
    out_dir_root = require("out_dir")
    base_dir = os.path.join(out_dir_root, today_str)
    out_dir = base_dir
    n_envs = require("n_envs")
    seed = require("seed")
    total_timesteps = require("total_timesteps")
    policy_cfg = require("policy")
    policy, policy_kwargs = resolve_policy_and_kwargs(policy_cfg, require("policy_kwargs"))
    env_kwargs = require("env_kwargs")
    os.makedirs(out_dir, exist_ok=True)
    # expose output dir to worker envs before envs are created
    try:
        os.environ['TRAIN_OUT_DIR'] = base_dir
    except Exception:
        pass
    ppo_param_keys = PPO_PARAM_KEYS
    # require all PPO params to be explicitly provided in config
    ppo_kwargs = {k: require(k) for k in ppo_param_keys}
    # Ensure numeric PPO params are proper types (yaml may load exponent as string)
    try:
        lr = ppo_kwargs.get("learning_rate")
        if isinstance(lr, str):
            try:
                ppo_kwargs["learning_rate"] = float(lr)
            except Exception:
                pass
    except Exception:
        pass
    # 读取评估/日志相关配置（统一以 optimization-step 为单位）
    save_iters = require("save_iters")
    summary_iters = require("summary_iters")
    # eval_every must be provided explicitly (optimization-step units)
    eval_every = require("eval_every")
    # EvalCallback.on_step 每个 VecEnv step 调用一次（n_envs 个环境同步前进），
    # 因此 eval_freq 的单位是 VecEnv step，而非环境步；1 optimization-step = n_steps 个 VecEnv step。
    n_steps = int(require("n_steps"))
    eval_freq = compute_eval_freq(eval_every, n_steps)
    n_eval_episodes = require("n_eval_episodes")
    eval_deterministic = require("eval_deterministic")
    eval_render = require("eval_render")
    tensorboard = require("tensorboard")
    tb_log_dirname = require("tb_log_dir")
    save_best_model = require("save_best_model")
    verbose_level = require("verbose")
    tb_log = os.path.join(base_dir, tb_log_dirname) if tensorboard else None
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger("train")
    logger.info("==== RL训练启动 ====")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"TensorBoard目录: {tb_log}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"环境名: {env_name}")
    logger.info(f"环境参数: {env_kwargs}")
    logger.info(f"训练配置: {cfg}")
    # Resolve the device before creating SC2 envs so a bad explicit device
    # fails fast instead of after spawning many worker processes.
    device = resolve_device(require("device"), logger=logger)
    logger.info(describe_device(device))
    vec_env = None
    eval_env = None
    try:
        vec_env = make_seeded_vec_env(env_name, env_kwargs, n_envs, seed)
        logger.info(
            f"训练环境 seed: {compute_env_seed(seed, 0)}..{compute_env_seed(seed, int(n_envs) - 1)} "
            f"(seed={seed}, n_envs={n_envs})"
        )
        # checkpoint_path is optional: if provided in the config it must be present as a key (can be empty to mean None)
        if "checkpoint_path" in cfg:
            checkpoint_path = cfg["checkpoint_path"]
            if checkpoint_path in (None, ""):
                checkpoint_path = None
        else:
            checkpoint_path = None
        checkpoint_loaded = bool(checkpoint_path and os.path.isfile(checkpoint_path))
        # SB3's internal TensorBoard writer stays enabled and writes its
        # default scalars under tb_log (SB3 may create a PPO_<n>/ subdirectory).
        # Our TBDualWriterCallback additionally writes project-specific
        # scalars into tb_log/extra; there is no separate `fundamental` writer.
        tb_log_for_sb3 = tb_log
        if checkpoint_loaded:
            logger.info(f"[INFO] 从checkpoint加载模型: {checkpoint_path}")
            # The saved policy class comes from the checkpoint; PPO.load's
            # `policy` kwarg only lands in __dict__ and does not replace it.
            model = PPO.load(checkpoint_path, env=vec_env, tensorboard_log=tb_log_for_sb3, policy_kwargs=policy_kwargs, device=device, **ppo_kwargs)
            # PPO.load does not apply a seed; re-seed the restored model RNGs so
            # training remains reproducible from the configured seed.
            model.set_random_seed(seed)
            logger.info(f"[INFO] 续训: 已恢复 num_timesteps={model.num_timesteps}，将追加步数而非重置计数。")
        else:
            # honor clip_value_loss config: when True, pass clip_range_vf to SB3 if supported
            clip_value_loss_flag = False
            if "clip_value_loss" in cfg:
                clip_value_loss_flag = cfg["clip_value_loss"]
                if clip_value_loss_flag:
                    # set clip_range_vf equal to clip_range (user-provided)
                    try:
                        ppo_kwargs["clip_range_vf"] = ppo_kwargs["clip_range"]
                    except Exception:
                        pass
            model = PPO(policy, vec_env, verbose=verbose_level, tensorboard_log=tb_log_for_sb3, policy_kwargs=policy_kwargs, device=device, seed=seed, **ppo_kwargs)
        # 使用EvalCallback只保存表现最好的模型
        best_model_save_path = out_dir if save_best_model else None
        class LogEvalCallback(EvalCallback):
            def __init__(self, *args, logger=None, save_best_only: bool = False, checkpoint_dir: str = None, **kwargs):
                super().__init__(*args, **kwargs)
                self._logger = logger or logging.getLogger("train")
                self._save_best_only = bool(save_best_only)
                self._checkpoint_dir = checkpoint_dir

            def _on_training_start(self) -> None:
                # run one immediate evaluation at training start to ensure eval metrics
                # are available early (useful for short smoke-runs)
                try:
                    super()._on_training_start()
                except Exception:
                    pass
                saved_n_calls = self.n_calls
                try:
                    # call EvalCallback's evaluation routine directly once
                    self._logger.info(f"[EvalCallback] 初始评估触发: eval_freq={self.eval_freq}")
                    # Temporarily use n_calls=0 so the immediate evaluation also runs
                    # when resuming with a non-zero counter; restore the resumed
                    # counter afterwards so the regular schedule keeps its phase.
                    self.n_calls = 0
                    self.num_timesteps = self.model.num_timesteps
                    try:
                        # this will call the internal evaluation logic and populate
                        # `last_mean_reward` / `evaluations_length` etc.
                        self._on_step()
                    except Exception:
                        pass
                except Exception:
                    pass
                finally:
                    self.n_calls = saved_n_calls

            def _on_step(self) -> bool:
                do_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
                if do_eval:
                    self._logger.info(
                        f"[EvalCallback] 触发评估: num_timesteps={self.num_timesteps}, n_calls={self.n_calls}, eval_freq={self.eval_freq}"
                    )
                # remember previous best to detect improvement after super
                prev_best = getattr(self, 'best_mean_reward', float('-inf'))
                result = super()._on_step()
                if do_eval:
                    self._logger.info(
                        f"[EvalCallback] 评估完成: num_timesteps={self.num_timesteps}, last_mean_reward={self.last_mean_reward}"
                    )
                    # expose latest eval stats for external callbacks (e.g. TB dual writer)
                    try:
                        eval_stats = {
                            "mean_reward": float(self.last_mean_reward),
                            "timesteps": int(self.num_timesteps),
                            "n_eval_episodes": int(self.n_eval_episodes),
                        }
                        try:
                            if getattr(self, 'evaluations_length', None) is not None and len(self.evaluations_length) > 0:
                                eval_stats["mean_ep_length"] = float(self.evaluations_length[-1].mean())
                        except Exception:
                            pass
                        setattr(self.model, '_last_eval_stats', eval_stats)
                    except Exception:
                        pass
                    # if configured to keep only best, and a new best was found, remove periodic checkpoints
                    try:
                        new_best = getattr(self, 'best_mean_reward', float('-inf'))
                        if self._save_best_only and self._checkpoint_dir is not None and new_best != prev_best:
                            # remove checkpoint_opt_*.zip files in checkpoint_dir
                            pattern = os.path.join(self._checkpoint_dir, 'checkpoint_opt_*.zip')
                            for path in glob.glob(pattern):
                                try:
                                    os.remove(path)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                return result

        eval_callback = None
        if eval_freq > 0:
            eval_seed_offset = compute_eval_seed_offset(n_envs)
            eval_env, eval_callback = build_evaluation(
                eval_freq,
                logger=logger,
                make_env=lambda: make_seeded_vec_env(
                    env_name, env_kwargs, 1, seed, offset=eval_seed_offset
                ),
                build_callback=lambda env: LogEvalCallback(
                    env,
                    best_model_save_path=best_model_save_path,
                    log_path=out_dir,
                    eval_freq=eval_freq,
                    deterministic=eval_deterministic,
                    render=eval_render,
                    n_eval_episodes=n_eval_episodes,
                    logger=logger,
                    save_best_only=save_best_model,
                    checkpoint_dir=out_dir,
                ),
            )
            logger.info(f"评估环境 seed: {compute_env_seed(seed, 0, eval_seed_offset)}")
            logger.info(
                f"[EvalCallback] 有效评估频率: eval_freq={eval_callback.eval_freq} VecEnv steps "
                f"= {eval_every} optimization-steps (n_envs={vec_env.num_envs}, n_steps={n_steps})"
            )
        else:
            logger.info("评估已禁用（eval_every<=0），仅保存最终模型。")
        logger.info(f"开始训练，总步数: {total_timesteps}")
        if eval_callback is not None:
            logger.info(f"评估频率(eval_freq): {eval_freq}, 每次评估episode数: {n_eval_episodes}, 保存最佳模型: {save_best_model}")
        else:
            logger.info("评估频率: 已禁用, 保存最佳模型: 不适用（仅保存最终模型）")
        logger.info(f"TensorBoard: {'启用' if tensorboard and tb_log else '禁用'}, TB目录: {tb_log}")
        # 日志/打印间隔（多少次学习更新写一次日志）
        log_interval = require("log_interval")
        logger.info(f"日志间隔(log_interval): {log_interval}")
        # Periodic checkpointing and TensorBoard are independent of evaluation.
        callbacks, tb_callback = build_callbacks(
            eval_callback=eval_callback,
            out_dir=out_dir,
            n_steps=n_steps,
            n_envs=vec_env.num_envs,
            save_iters=save_iters,
            summary_iters=summary_iters,
            tensorboard=tensorboard,
            tb_log=tb_log,
        )

        if checkpoint_loaded:
            resume_offsets = compute_resume_offsets(
                model.num_timesteps, vec_env.num_envs, n_steps
            )
            if eval_callback is not None:
                eval_callback.n_calls = resume_offsets["vec_steps"]
            if tb_callback is not None:
                tb_callback.opt_step = resume_offsets["opt_steps"]
            eval_calls = eval_callback.n_calls if eval_callback is not None else None
            logger.info(
                f"续训计数偏移: eval n_calls={eval_calls}, "
                f"opt_step={resume_offsets['opt_steps']}"
            )
        callback_list = CallbackList(callbacks)

        # If total_timesteps <= 0, treat as "run until interrupted"; otherwise run for given timesteps
        try:
            tt = int(total_timesteps) if total_timesteps is not None else None
        except Exception:
            tt = None

        if tt is None:
            # if unspecified, require explicit total_timesteps in config
            raise ValueError("total_timesteps must be specified in config and be a non-negative integer")

        if tt <= 0:
            logger.info("运行模式: total_timesteps <= 0，持续训练直到手动中断 (Ctrl+C)。")
            first_learn = True
            try:
                while True:
                    # A fresh model keeps SB3's default reset semantics; a loaded
                    # checkpoint must continue its counters instead of restarting.
                    reset_this_learn = (not checkpoint_loaded) and first_learn
                    target = compute_learn_target(
                        10**9, model.num_timesteps, reset_this_learn
                    )
                    first_learn = False
                    model.learn(
                        total_timesteps=target,
                        callback=callback_list,
                        reset_num_timesteps=reset_this_learn,
                        log_interval=log_interval,
                    )
            except KeyboardInterrupt:
                logger.info("检测到 KeyboardInterrupt，停止训练并保存。")
        else:
            reset_this_learn = not checkpoint_loaded
            target = compute_learn_target(tt, model.num_timesteps, reset_this_learn)
            logger.info(
                f"learn 目标步数: {target} (请求追加 {tt}, 当前 {model.num_timesteps}, "
                f"reset_num_timesteps={reset_this_learn})"
            )
            model.learn(
                total_timesteps=target,
                callback=callback_list,
                reset_num_timesteps=reset_this_learn,
                log_interval=log_interval,
            )
        logger.info("训练完成，保存最终模型...")
        model.save(os.path.join(out_dir, f"final_model_{today_str}"))
        resolved = {"policy": policy}
        if "features_extractor_class" in policy_kwargs:
            resolved["features_extractor_class"] = policy_kwargs["features_extractor_class"]
        snapshot = build_config_snapshot(cfg, resolved, DEFAULT_CONFIG_PATH)
        with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            f.write("# Auto-generated by train.py. Config keys below are a rerunnable snapshot;\n")
            f.write("# runtime classes are recorded under _snapshot and are not passed to PPO.\n")
            yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
        logger.info("配置已保存: config_used.yaml")
        # If configured to only keep the best model, remove periodic checkpoint
        # files created during training. Only when evaluation (and therefore
        # best-model saving) was actually enabled; otherwise they are the only
        # intermediate artifacts and must be kept.
        try:
            if should_prune_periodic_checkpoints(save_best_model, eval_callback is not None):
                pattern = os.path.join(out_dir, 'checkpoint_opt_*.zip')
                for path in glob.glob(pattern):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                logger.info("已清理周期性 checkpoint，只保留最佳/最终模型（save_best_model=True）。")
        except Exception:
            pass
    finally:
        # Always release SC2 processes; close errors must not mask a training
        # exception. Training output is intentionally left in place for the
        # caller (smoke_run.py) to remove.
        for env_label, env in (("eval_env", eval_env), ("vec_env", vec_env)):
            if env is None:
                continue
            try:
                env.close()
            except Exception as exc:
                logger.warning(f"关闭 {env_label} 失败: {exc}")

if __name__ == "__main__":
    # 修复 pysc2 的 shuffled_hue 问题，兼容 Python 3.9+
    import sys
    import types

    def shuffled_hue_patch(palette_size):
        import random
        palette = [i for i in range(palette_size)]
        random.shuffle(palette)  # Python 3.9+ 兼容写法
        return palette

    # 猴子补丁 pysc2 的 shuffled_hue
    sys.modules_backup = dict(sys.modules)
    try:
        import pysc2.lib.colors
        pysc2.lib.colors.shuffled_hue = shuffled_hue_patch
    except ImportError:
        pass

    main()
