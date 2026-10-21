"""Device resolution and required-config-key tests (no SC2 launch)."""
import ast
import inspect
import os
import sys
import textwrap

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import train as train_module
from train import REQUIRED_CONFIG_KEYS, resolve_device

CONFIG_PATHS = [
    os.path.join(REPO_ROOT, "configs", "ppo_config.yaml"),
    os.path.join(REPO_ROOT, "configs", "bitahub_first_run.yaml"),
    os.path.join(REPO_ROOT, "configs", "bitahub_smoke.yaml"),
]


def test_auto_uses_cuda_when_available():
    assert resolve_device("auto", cuda_available=True) == "cuda"


def test_auto_falls_back_to_cpu_without_cuda():
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_missing_or_empty_device_defaults_to_auto():
    assert resolve_device(None, cuda_available=False) == "cpu"
    assert resolve_device("", cuda_available=True) == "cuda"


def test_explicit_cpu_is_honored_even_with_cuda():
    assert resolve_device("cpu", cuda_available=True) == "cpu"


def test_explicit_cuda_is_accepted_when_available():
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    assert resolve_device("cuda", cuda_available=True, cuda_device_count=1) == "cuda"
    assert resolve_device("cuda:0", cuda_available=True, cuda_device_count=1) == "cuda:0"


def test_cuda_index_must_be_within_device_count():
    with pytest.raises(ValueError, match="out of range"):
        resolve_device("cuda:1", cuda_available=True, cuda_device_count=1)
    assert resolve_device("cuda:1", cuda_available=True, cuda_device_count=2) == "cuda:1"


def test_explicit_cuda_fails_fast_when_unavailable():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda:0", cuda_available=False)


def test_invalid_device_values_are_rejected():
    with pytest.raises(ValueError):
        resolve_device("gpu", cuda_available=True)
    with pytest.raises(ValueError):
        resolve_device("cuda:abc", cuda_available=True)
    with pytest.raises(ValueError):
        resolve_device("cuda:-1", cuda_available=True)


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_configs_contain_required_keys(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in cfg]
    assert not missing, f"{config_path} is missing required keys: {missing}"


def test_vec_env_cls_is_required_without_default():
    """The VecEnv type trades memory for throughput; no silent default allowed."""
    assert "vec_env_cls" in REQUIRED_CONFIG_KEYS
    signature = inspect.signature(train_module.make_seeded_vec_env)
    assert signature.parameters["vec_env_cls"].default is inspect.Parameter.empty


def test_missing_vec_env_cls_fails_fast():
    with pytest.raises(ValueError, match="vec_env_cls"):
        train_module.resolve_vec_env_cls(None)


def test_vec_env_cls_is_normalized_and_validated():
    assert train_module.resolve_vec_env_cls(" Dummy ") == "dummy"
    assert train_module.resolve_vec_env_cls("subproc") == "subproc"
    with pytest.raises(ValueError, match="vec_env_cls"):
        train_module.resolve_vec_env_cls("serial")


def test_default_and_bitahub_configs_declare_device():
    for name in ("ppo_config.yaml", "bitahub_first_run.yaml"):
        path = os.path.join(REPO_ROOT, "configs", name)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "device" in cfg, f"{name} should declare a device field"


def test_checkpoint_load_does_not_pass_policy_class():
    """PPO.load restores the saved policy class; a `policy=` kwarg is ignored."""
    source = textwrap.dedent(inspect.getsource(train_module.main))
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PPO"
    ]
    assert loads, "expected a PPO.load(...) call in train.main"
    for node in loads:
        policy_kwargs = [kw.arg for kw in node.keywords]
        assert "policy" not in policy_kwargs, "PPO.load does not accept a policy class override"
        assert "policy_kwargs" in policy_kwargs
