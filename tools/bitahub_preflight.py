#!/usr/bin/env python3
"""Bitahub preflight checks.

Read-only: never launches StarCraft II, installs packages, or touches the lock.
Exit code 0 means no blocking issue; exit code 1 means at least one FAIL.
WARN entries do not affect the exit code.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    print(f"[FAIL] import yaml: {type(exc).__name__}: {exc}")
    sys.exit(1)

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "bitahub_first_run.yaml"

# Keys read with require() in train.py. checkpoint_path is optional there.
REQUIRED_KEYS = [
    "env", "out_dir", "n_envs", "seed", "device", "policy", "policy_kwargs", "env_kwargs",
    "total_timesteps", "learning_rate", "ent_coef", "batch_size", "n_epochs",
    "gamma", "gae_lambda", "n_steps", "clip_range", "vf_coef", "max_grad_norm",
    "save_iters", "summary_iters", "eval_every", "n_eval_episodes",
    "eval_deterministic", "eval_render", "tensorboard", "tb_log_dir",
    "log_interval", "save_best_model", "verbose",
]


class Report:
    def __init__(self):
        self.fails = []
        self.warns = []

    def _line(self, status, name, detail=""):
        line = f"[{status:<4}] {name}"
        if detail:
            line += f": {detail}"
        print(line)

    def ok(self, name, detail=""):
        self._line(PASS, name, detail)

    def warn(self, name, detail=""):
        self.warns.append(name)
        self._line(WARN, name, detail)

    def fail(self, name, detail=""):
        self.fails.append(name)
        self._line(FAIL, name, detail)

    def skip(self, name, detail=""):
        self._line(SKIP, name, detail)


def check_import(rep, module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        rep.fail(f"import {module_name}", f"{type(exc).__name__}: {exc}")
        return None
    version = getattr(module, "__version__", "unknown")
    rep.ok(f"import {module_name}", str(version))
    return module


def check_python(rep):
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] == (3, 10):
        rep.ok("Python", version)
    else:
        rep.fail("Python", f"{version}（项目锁定 3.10.x）")


def check_cuda(rep, torch):
    if torch is None:
        return
    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build:
        rep.ok("torch CUDA build", f"cu{cuda_build}")
    else:
        rep.fail("torch CUDA build", "torch.version.cuda=None（CPU-only wheel；Linux 应装 cu121 构建）")

    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        rep.fail("torch.cuda.is_available()", f"{type(exc).__name__}: {exc}")
        return
    if not available:
        rep.fail("CUDA 可用", "torch.cuda.is_available()=False（硬件/驱动缺失，不是脚本错误）")
        return
    count = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(count))
    rep.ok("CUDA 可用", f"device_count={count}; {names}")


def check_nvidia_smi(rep):
    if not sys.platform.startswith("linux"):
        rep.skip("nvidia-smi", f"仅 Linux 检查（当前 {sys.platform}）")
        return
    smi = shutil.which("nvidia-smi")
    if smi is None:
        rep.warn("nvidia-smi", "未找到；GPU 实例请确认驱动已安装")
        return
    cmd = [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        rep.warn("nvidia-smi", f"{type(exc).__name__}: {exc}")
        return
    output = (result.stdout or "").strip()
    if result.returncode == 0 and output:
        for line in output.splitlines():
            rep.ok("nvidia-smi", line.strip())
    else:
        rep.warn("nvidia-smi", (result.stderr or "无输出").strip())


def check_sc2(rep):
    env_path = os.environ.get("SC2PATH")
    if env_path:
        sc2path = Path(env_path).expanduser()
        rep.ok("SC2PATH", str(sc2path))
    else:
        sc2path = Path.home() / "StarCraftII"
        rep.warn("SC2PATH", f"未设置，将使用 PySC2 默认路径 {sc2path}")

    if not sc2path.is_dir():
        # 检查常见自定义路径
        common_paths = [Path("/mnt/ppo/StarCraftII"), Path("/opt/StarCraftII")]
        found = next((p for p in common_paths if p.is_dir()), None)
        if found:
            rep.fail(
                "SC2 目录",
                f"不存在: {sc2path}，但发现 {found}。"
                f"请执行: export SC2PATH={found} 或 source scripts/env.sh"
            )
        else:
            rep.fail("SC2 目录", f"不存在: {sc2path}（需放置 SC2 本体或导出 SC2PATH）")
        return

    if sys.platform.startswith("win"):
        candidates = [sc2path / "SC2_x64.exe"]
        candidates += sorted(sc2path.glob("Versions/Base*/SC2_x64.exe"))
    elif sys.platform == "darwin":
        candidates = [sc2path / "SC2.app"]
    else:
        candidates = [sc2path / "SC2_x64", sc2path / "Support64" / "SC2_x64"]
        candidates += sorted(sc2path.glob("Versions/Base*/SC2_x64"))
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        rep.fail("SC2 可执行文件", f"未找到 {sc2path / 'SC2_x64'}（Windows 为 SC2_x64.exe）")
    else:
        rep.ok("SC2 可执行文件", str(binary))

    # PySC2 mini-games live under Maps/mini_games/; some installs put them
    # directly under Maps/. Check both, plus explicit lowercase variants.
    map_candidates = []
    for maps_dir in ("Maps", "maps"):
        for sub_dir in ("mini_games", ""):
            candidate = sc2path / maps_dir / sub_dir / "MoveToBeacon.SC2Map"
            if candidate not in map_candidates:
                map_candidates.append(candidate)
    map_path = next((path for path in map_candidates if path.is_file()), None)
    if map_path is not None:
        rep.ok("MoveToBeacon 地图", str(map_path))
    else:
        checked = "; ".join(str(path) for path in map_candidates)
        rep.fail("MoveToBeacon 地图", f"未找到；已检查: {checked}")


def check_display(rep):
    if not sys.platform.startswith("linux"):
        rep.skip("DISPLAY", f"仅 Linux 检查（当前 {sys.platform}）")
        return
    display = os.environ.get("DISPLAY")
    if not display:
        # 检查 Xvfb 是否在运行
        xvfb_running = False
        try:
            result = subprocess.run(
                ["pgrep", "-x", "Xvfb"], capture_output=True, text=True, timeout=5
            )
            xvfb_running = result.returncode == 0
        except Exception:
            pass

        if xvfb_running:
            rep.fail(
                "DISPLAY",
                "未设置，但检测到 Xvfb 在运行。"
                "请执行: export DISPLAY=:99 或 source scripts/env.sh"
            )
        else:
            rep.fail(
                "DISPLAY",
                "未设置；SC2 需要 X 显示。建议先执行: "
                "Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99"
            )
        if shutil.which("Xvfb") is None:
            rep.warn("Xvfb", "未找到 Xvfb；无头实例需先安装/提供 X server")
        return

    # DISPLAY being set only means the variable exists; it does not prove the
    # X server is reachable or that GL/SC2 can start.
    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        rep.warn(
            "DISPLAY 可见性",
            f"DISPLAY={display} 已设置，但未找到 xdpyinfo，无法验证 X server；"
            "存在不等于 GL/SC2 可启动",
        )
    else:
        try:
            result = subprocess.run(
                [xdpyinfo, "-display", display], capture_output=True, text=True, timeout=10
            )
        except Exception as exc:
            rep.warn("DISPLAY 可见性", f"xdpyinfo 执行失败: {type(exc).__name__}: {exc}")
        else:
            if result.returncode == 0:
                rep.ok("DISPLAY 可见性", f"{display} 可连接（仍需确认 GL/SC2 可启动）")
            else:
                detail = (result.stderr or "无输出").strip().splitlines()[0]
                rep.warn(
                    "DISPLAY 可见性",
                    f"DISPLAY={display} 无法连接: {detail}；存在不等于 GL/SC2 可启动",
                )


def positive_int(rep, cfg, key):
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        rep.fail(key, f"必须为 >0 的整数，当前为 {value!r}")
        return None
    rep.ok(key, str(value))
    return value


def check_device_config(rep, cfg, torch):
    """Validate the optional `device` key the same way train.resolve_device does.

    CUDA being unavailable is an environment issue, not a script bug: only an
    invalid value is a config FAIL here.
    """
    if "device" not in cfg:
        rep.ok("device", "未设置；train.py 默认 auto（CUDA 可用则 cuda，否则 cpu）")
        return
    raw = cfg["device"]
    value = "auto" if raw is None else str(raw).strip().lower()
    if not value:
        value = "auto"

    if value in ("auto", "cpu"):
        rep.ok("device", value)
        return

    if value == "cuda" or value.startswith("cuda:"):
        index = None
        if ":" in value:
            index_text = value.split(":", 1)[1]
            if not index_text.isdigit():
                rep.fail("device", f"非法 CUDA 索引 {raw!r}；应为 cuda:<非负整数>")
                return
            index = int(index_text)
        if torch is None:
            rep.warn("device", f"{value} 要求 CUDA，但 torch 导入失败，无法确认")
            return
        try:
            available = torch.cuda.is_available()
        except Exception as exc:
            rep.warn("device", f"无法检查 CUDA: {type(exc).__name__}: {exc}")
            return
        if not available:
            rep.warn(
                "device",
                f"配置为 {value} 但 CUDA 不可用；train.py 会 fail fast。"
                "这是环境/配置问题（或改用 auto/cpu），不是脚本实现错误",
            )
            return
        if index is not None:
            try:
                count = int(torch.cuda.device_count())
            except Exception as exc:
                rep.warn(
                    "device",
                    f"无法获取 CUDA 设备数量，不能校验 {value}: {type(exc).__name__}: {exc}",
                )
                return
            if index >= count:
                rep.fail(
                    "device",
                    f"{value} 超出设备范围：torch.cuda.device_count()={count}"
                    f"（有效索引 0..{count - 1}）",
                )
                return
        rep.ok("device", value)
        return

    rep.fail("device", f"不支持 {raw!r}；应为 auto / cpu / cuda / cuda:<index>")


def check_config(rep, config_path, torch=None):
    path = Path(config_path).expanduser()
    if not path.is_file():
        rep.fail("配置文件", f"不存在: {path}")
        return
    rep.ok("配置文件", str(path))

    try:
        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except Exception as exc:
        rep.fail("YAML 解析", f"{type(exc).__name__}: {exc}")
        return
    if not isinstance(cfg, dict):
        rep.fail("YAML 结构", "顶层必须是映射（dict）")
        return

    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        rep.fail("配置必要键", "缺少: " + ", ".join(missing))
    else:
        rep.ok("配置必要键", f"{len(REQUIRED_KEYS)} 项齐全")
    if "eval_freq" in cfg:
        rep.fail("legacy key eval_freq", "train.py 会拒绝该键，请使用 eval_every")

    total_timesteps = cfg.get("total_timesteps")
    if (isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int)
            or total_timesteps <= 0):
        rep.fail(
            "total_timesteps",
            f"必须为 >0 的整数，当前为 {total_timesteps!r}（<=0 表示无限训练，云端不可用）",
        )
    else:
        rep.ok("total_timesteps", str(total_timesteps))

    n_envs = positive_int(rep, cfg, "n_envs")
    n_steps = positive_int(rep, cfg, "n_steps")
    batch_size = positive_int(rep, cfg, "batch_size")
    if n_envs is not None and n_steps is not None and batch_size is not None:
        rollout = n_envs * n_steps
        if batch_size <= rollout:
            rep.ok("batch_size <= n_envs*n_steps", f"{batch_size} <= {rollout}")
        else:
            rep.fail("batch_size <= n_envs*n_steps", f"{batch_size} > {rollout}")

    check_device_config(rep, cfg, torch)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bitahub 启动前只读检查（不会启动 SC2）")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="训练配置 YAML（默认 configs/bitahub_first_run.yaml）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rep = Report()
    print("== Bitahub preflight ==")
    print(f"配置: {args.config}")
    print()

    check_python(rep)
    torch = check_import(rep, "torch")
    check_import(rep, "stable_baselines3")
    check_import(rep, "pysc2")
    # The wrapper actually used by train.py and the training entry points.
    check_import(rep, "pysc2.env.sc2_env")
    check_import(rep, "tensorboard")
    check_cuda(rep, torch)
    check_nvidia_smi(rep)
    check_sc2(rep)
    check_display(rep)
    check_config(rep, args.config, torch)

    print()
    print(f"结果: {len(rep.fails)} FAIL, {len(rep.warns)} WARN")
    if rep.fails:
        print("存在 FAIL，不能开始训练。")
        return 1
    print("Preflight 通过；WARN 项请人工确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
