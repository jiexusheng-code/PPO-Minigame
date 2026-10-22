"""Tests for the tool scripts (no SC2 launch, no training)."""
import copy
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from train import REQUIRED_CONFIG_KEYS, compute_eval_freq

BITAHUB_CONFIG = os.path.join(REPO_ROOT, "configs", "bitahub_first_run.yaml")
COLLECT_MINERAL_SHARDS_CONFIGS = [
    os.path.join(REPO_ROOT, "configs", name)
    for name in (
        "collect_mineral_shards.yaml",
        "collect_mineral_shards_smoke.yaml",
    )
]


def _load_tool(name):
    path = os.path.join(REPO_ROOT, "tools", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_tool("bitahub_preflight")
smoke_run = _load_tool("smoke_run")


# ---------------------------------------------------------------- preflight

def test_preflight_required_keys_match_train():
    assert set(preflight.REQUIRED_KEYS) == set(REQUIRED_CONFIG_KEYS)


def test_preflight_accepts_bitahub_config():
    rep = preflight.Report()
    preflight.check_config(rep, BITAHUB_CONFIG)
    assert not rep.fails, rep.fails


def test_preflight_supports_collect_mineral_shards():
    assert "CollectMineralShards" in preflight.SUPPORTED_MAPS


@pytest.mark.parametrize("config_path", COLLECT_MINERAL_SHARDS_CONFIGS)
def test_preflight_accepts_collect_mineral_shards_configs(config_path):
    rep = preflight.Report()
    preflight.check_config(rep, config_path)
    assert not rep.fails, rep.fails


@pytest.mark.parametrize("config_path", COLLECT_MINERAL_SHARDS_CONFIGS)
def test_collect_mineral_shards_configs_use_stochastic_eval(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["env"] == "CollectMineralShards"
    assert cfg["eval_deterministic"] is False
    if config_path.endswith("_smoke.yaml"):
        assert cfg["n_envs"] == 1
        assert cfg["vec_env_cls"] == "dummy"
        assert cfg["total_timesteps"] == 256
        assert cfg["eval_every"] == 0


def _bitahub_cfg(**overrides):
    with open(BITAHUB_CONFIG, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.update(overrides)
    return cfg


def test_preflight_rejects_invalid_device(tmp_path):
    path = tmp_path / "bad_device.yaml"
    path.write_text(yaml.safe_dump(_bitahub_cfg(device="gpu")), encoding="utf-8")
    rep = preflight.Report()
    preflight.check_config(rep, path)
    assert "device" in rep.fails


def test_preflight_flags_missing_required_key(tmp_path):
    cfg = _bitahub_cfg()
    cfg.pop("eval_every")
    path = tmp_path / "missing_key.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rep = preflight.Report()
    preflight.check_config(rep, path)
    assert "配置必要键" in rep.fails


def test_preflight_explicit_cuda_without_cuda_is_warn_not_fail(tmp_path):
    path = tmp_path / "cuda_device.yaml"
    path.write_text(yaml.safe_dump(_bitahub_cfg(device="cuda")), encoding="utf-8")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    rep = preflight.Report()
    preflight.check_config(rep, path, fake_torch)
    assert "device" not in rep.fails
    assert "device" in rep.warns


def test_preflight_explicit_cuda_with_cuda_passes(tmp_path):
    path = tmp_path / "cuda_device.yaml"
    path.write_text(yaml.safe_dump(_bitahub_cfg(device="cuda:0")), encoding="utf-8")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    rep = preflight.Report()
    preflight.check_config(rep, path, fake_torch)
    assert not rep.fails, rep.fails


def test_preflight_rejects_cuda_index_out_of_range(tmp_path):
    path = tmp_path / "cuda_out_of_range.yaml"
    path.write_text(yaml.safe_dump(_bitahub_cfg(device="cuda:1")), encoding="utf-8")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    rep = preflight.Report()
    preflight.check_config(rep, path, fake_torch)
    assert "device" in rep.fails


# ---------------------------------------------------------------- smoke run

def test_smoke_config_disables_eval_and_keeps_smoke_budget():
    with open(os.path.join(REPO_ROOT, "configs", "ppo_config.yaml"), "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    original = copy.deepcopy(base)

    cfg = smoke_run.build_smoke_config(base)

    assert cfg["eval_every"] == 0
    # eval_every=0 -> eval_freq=0 -> SB3 EvalCallback never evaluates,
    # including train.py's custom initial _on_step() call.
    assert compute_eval_freq(cfg["eval_every"], cfg["n_steps"]) == 0
    assert cfg["total_timesteps"] == 512
    assert cfg["n_envs"] == 1
    assert cfg["env_kwargs"]["visualize"] is False
    assert cfg["batch_size"] <= cfg["n_envs"] * cfg["n_steps"]
    # build_smoke_config must not mutate the production config.
    assert base == original
