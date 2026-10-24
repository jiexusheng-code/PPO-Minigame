"""Pre-update critic diagnostics for PPO rollouts.

The diagnostics in this module are written at ``Callback.on_rollout_end``,
which SB3 calls *after* GAE targets are computed and *before* ``PPO.train()``.
All scalars therefore describe the just-collected rollout and the policy that
produced it; none of them include an optimization-step update.

Two kinds of evidence are produced:

1. Rollout-level summaries (reward/value/return/advantage scale, residual
   error, correlation and explained variance) plus a bounded (<=32 sample)
   probe of the feature extractor and value MLP.
2. On a sparse cadence (first complete rollout, then every 50k environment
   steps) one compressed ``.npz`` sample plus a JSON schema/metadata sidecar
   and a sampled value-only gradient probe.

Read-only guarantees: the rollout buffer is read directly (never
``buffer.get``, so ``generator_ready`` is untouched), probe hooks are
temporary, the extractor's spatial-map cache is cleared in ``finally``, and
the gradient probe uses ``torch.autograd.grad`` (never ``backward``) so no
``param.grad``, optimizer state, or RNG state changes.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Scalar tag prefix shared with TBDualWriterCallback.
CRITIC_DIAG_PREFIX = "critic/rollout_preupdate"
CRITIC_SATURATION_SAMPLE_SIZE = 32
CRITIC_TANH_SATURATION_ABS = 0.999
CRITIC_VALUE_ABS_GE_1 = 1.0
CRITIC_VALUE_ABS_GE_10 = 10.0
ZERO_VARIANCE_EPS = 1e-12

DIAGNOSTICS_DIRNAME = "diagnostics"
DIAG_SAMPLE_INTERVAL_ENV_STEPS = 50_000
DIAG_SAMPLE_SCHEMA_VERSION = 1

_GRAD_PREFIX = f"{CRITIC_DIAG_PREFIX}/grad_preupdate"


def _saturation_threshold_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _time_major_flat(array) -> Optional[np.ndarray]:
    """Flatten ``(n_steps, n_envs, *rest)`` to ``(t * n_envs + e, *rest)``.

    Only the pre-``train()`` layout is accepted: SB3's ``get()`` flattens with
    ``swapaxes(0, 1)`` (env-major), which cannot be relabelled as time-major.
    """
    arr = np.asarray(array)
    if arr.ndim < 2:
        return None
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def _rollout_layout(buffer) -> Optional[Tuple[int, int, int]]:
    """Return ``(transitions, n_steps, n_envs)`` before ``train()`` flattening."""
    values = getattr(buffer, "values", None)
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim < 2 or arr.size == 0:
        return None
    n_steps, n_envs = int(arr.shape[0]), int(arr.shape[1])
    if n_steps <= 0 or n_envs <= 0:
        return None
    return n_steps * n_envs, n_steps, n_envs


def _summary_stats(name: str, flat: np.ndarray) -> Dict[str, float]:
    """Mean/std(ddof=0)/min/max over finite entries plus finite fraction."""
    values = np.asarray(flat, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean()) if values.size else 0.0
    stats = {f"{name}_finite_fraction": finite_fraction}
    if not finite.any():
        return stats
    good = values[finite]
    stats[f"{name}_mean"] = float(good.mean())
    stats[f"{name}_std_ddof0"] = float(good.std(ddof=0))
    stats[f"{name}_min"] = float(good.min())
    stats[f"{name}_max"] = float(good.max())
    return stats


def _residual_stats(values: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    """Residual (return - value) error, correlation and explained variance.

    The zero-variance boundary is reported explicitly instead of emitting a
    NaN/Inf: when either series or the return target has (near) zero variance,
    the correlation/EV scalar is set to 0.0 and the corresponding
    ``*_zero_variance`` flag is set to 1.0.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if v.shape != r.shape:
        return {}
    mask = np.isfinite(v) & np.isfinite(r)
    stats = {"residual_finite_fraction": float(mask.mean()) if mask.size else 0.0}
    if not mask.any():
        return stats
    v = v[mask]
    r = r[mask]
    residual = r - v
    stats["residual_mean"] = float(residual.mean())
    stats["residual_std_ddof0"] = float(residual.std(ddof=0))
    stats["residual_mse"] = float(np.mean(residual * residual))

    v_std = float(v.std(ddof=0))
    r_std = float(r.std(ddof=0))
    if v_std <= ZERO_VARIANCE_EPS or r_std <= ZERO_VARIANCE_EPS:
        stats["value_return_corr"] = 0.0
        stats["value_return_corr_zero_variance"] = 1.0
    else:
        corr = float(np.corrcoef(v, r)[0, 1])
        if math.isfinite(corr):
            stats["value_return_corr"] = corr
            stats["value_return_corr_zero_variance"] = 0.0
        else:
            stats["value_return_corr"] = 0.0
            stats["value_return_corr_zero_variance"] = 1.0

    target_var = float(np.var(r, ddof=0))
    if target_var <= ZERO_VARIANCE_EPS:
        stats["explained_variance"] = 0.0
        stats["explained_variance_zero_variance"] = 1.0
    else:
        ev = 1.0 - float(np.var(residual, ddof=0)) / target_var
        if math.isfinite(ev):
            stats["explained_variance"] = ev
            stats["explained_variance_zero_variance"] = 0.0
        else:
            stats["explained_variance"] = 0.0
            stats["explained_variance_zero_variance"] = 1.0
    return stats


def compute_rollout_stats(buffer) -> Dict[str, float]:
    """Compute pre-update summaries over the whole rollout buffer."""
    stats: Dict[str, float] = {}
    if bool(getattr(buffer, "generator_ready", False)):
        logger.warning(
            "critic 诊断: buffer 已被 get() 按 env-major 展平，无法还原 time-major 索引，跳过统计"
        )
        return stats
    fields = (
        ("reward", "rewards"),
        ("value", "values"),
        ("return", "returns"),
        ("advantage", "advantages"),
    )
    flat: Dict[str, np.ndarray] = {}
    for name, attr in fields:
        raw = getattr(buffer, attr, None)
        if raw is None:
            continue
        values = _time_major_flat(raw)
        if values is None:
            continue
        values = values.reshape(-1)
        flat[name] = values
        stats.update(_summary_stats(name, values))

    if "value" in flat and "return" in flat:
        stats.update(_residual_stats(flat["value"], flat["return"]))

    episode_starts = _time_major_flat(getattr(buffer, "episode_starts", None))
    if episode_starts is not None:
        starts = episode_starts.reshape(-1)
        if starts.size:
            stats["episode_start_fraction"] = float((starts != 0).mean())
            stats["episode_start_count"] = float((starts != 0).sum())

    layout = _rollout_layout(buffer)
    if layout is not None:
        total, num_steps, num_envs = layout
        stats["rollout_transition_count"] = float(total)
        stats["rollout_num_steps"] = float(num_steps)
        stats["rollout_num_envs"] = float(num_envs)
    return stats


@dataclass
class RolloutSample:
    """A deterministic, time/env-aligned slice of the rollout buffer."""

    indices_flat: np.ndarray
    indices_time: np.ndarray
    indices_env: np.ndarray
    observations: Dict[str, np.ndarray]
    values: Optional[np.ndarray]
    returns: Optional[np.ndarray]
    advantages: Optional[np.ndarray]
    rewards: Optional[np.ndarray]
    episode_starts: Optional[np.ndarray]
    num_steps: int
    num_envs: int
    transition_count: int
    generator_ready: bool


def _sample_indices(total: int, sample_size: int) -> np.ndarray:
    """Evenly spaced flat indices covering both the first and last transition.

    Including the last transition avoids the periodic-env bias of
    ``linspace(0, total, count, endpoint=False)``: with a 64x16 rollout that
    variant sampled only env 0. ``linspace(0, total - 1, count)`` spreads
    samples across all envs and the late part of the rollout.
    """
    count = int(min(max(1, sample_size), total))
    if count >= total:
        return np.arange(total, dtype=np.int64)
    return np.round(np.linspace(0, total - 1, count)).astype(np.int64)


def sample_rollout(buffer, sample_size: int = CRITIC_SATURATION_SAMPLE_SIZE) -> Optional[RolloutSample]:
    """Take a bounded sample without calling ``buffer.get``.

    Only the observation dict is required; value/return/advantage/reward
    arrays are included when the buffer provides them.
    """
    if bool(getattr(buffer, "generator_ready", False)):
        logger.warning(
            "critic 诊断: buffer 已被 get() 按 env-major 展平，无法还原 time-major 索引，跳过采样"
        )
        return None
    layout = _rollout_layout(buffer)
    if layout is None:
        return None
    total, num_steps, num_envs = layout
    indices = _sample_indices(total, sample_size)

    raw_obs = getattr(buffer, "observations", None)
    if not isinstance(raw_obs, dict) or not raw_obs:
        return None
    observations: Dict[str, np.ndarray] = {}
    for key, values in raw_obs.items():
        flat = _time_major_flat(values)
        if flat is None or flat.shape[0] != total:
            continue
        if not np.issubdtype(flat.dtype, np.number):
            continue
        observations[str(key)] = np.array(flat[indices], copy=True)
    if not observations:
        return None

    def _take(attr: str) -> Optional[np.ndarray]:
        flat_values = _time_major_flat(getattr(buffer, attr, None))
        if flat_values is None or flat_values.shape[0] != total:
            return None
        return np.array(flat_values.reshape(-1)[indices], copy=True)

    return RolloutSample(
        indices_flat=indices,
        indices_time=(indices // num_envs).astype(np.int64),
        indices_env=(indices % num_envs).astype(np.int64),
        observations=observations,
        values=_take("values"),
        returns=_take("returns"),
        advantages=_take("advantages"),
        rewards=_take("rewards"),
        episode_starts=_take("episode_starts"),
        num_steps=num_steps,
        num_envs=num_envs,
        transition_count=total,
        generator_ready=False,
    )


@dataclass
class ProbeResult:
    sample_count: int
    values: np.ndarray
    scalars: Dict[str, float]
    per_sample: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    tanh_saturation: List[float] = field(default_factory=list)


def _module_stats(
    name: str,
    tensor: torch.Tensor,
    saturation_abs: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compact scalar + per-sample statistics for one activation tensor.

    Returns only small (B,) arrays; full activations are never retained.
    """
    x = tensor.detach()
    if x.dim() == 0:
        x = x.reshape(1)
    x = x.reshape(x.shape[0], -1).float()
    finite = torch.isfinite(x)
    numel = int(x.numel())
    per_sample_numel = max(1, x.shape[1])
    finite_count = int(finite.sum().item())
    scalars: Dict[str, float] = {}
    per_sample: Dict[str, np.ndarray] = {}

    finite_fraction = float(finite_count / numel) if numel else 0.0
    scalars[f"act/{name}/finite_fraction"] = finite_fraction

    safe = torch.where(finite, x, torch.zeros_like(x))
    counts = finite.sum(dim=1).clamp_min(1)
    sample_mean = safe.sum(dim=1) / counts
    centered = torch.where(finite, x - sample_mean[:, None], torch.zeros_like(x))
    sample_var = (centered * centered).sum(dim=1) / counts
    sample_std = torch.sqrt(sample_var.clamp_min(0.0))
    sample_abs_max = torch.where(finite, x.abs(), torch.zeros_like(x)).amax(dim=1)
    nonfinite_fraction = 1.0 - finite.sum(dim=1).float() / per_sample_numel

    per_sample[f"{name}_per_sample_mean"] = sample_mean.cpu().numpy().astype(np.float32)
    per_sample[f"{name}_per_sample_std"] = sample_std.cpu().numpy().astype(np.float32)
    per_sample[f"{name}_per_sample_abs_max"] = sample_abs_max.cpu().numpy().astype(np.float32)
    per_sample[f"{name}_per_sample_nonfinite_fraction"] = (
        nonfinite_fraction.cpu().numpy().astype(np.float32)
    )

    if finite_count:
        finite_values = x[finite]
        scalars[f"act/{name}/abs_max"] = float(finite_values.abs().max().item())
        scalars[f"act/{name}/mean_abs"] = float(finite_values.abs().mean().item())
        scalars[f"act/{name}/mean"] = float(finite_values.mean().item())

    finite_rows = finite.any(dim=1)
    if finite_rows.any():
        scalars[f"act/{name}/mean_per_sample_std"] = float(
            sample_std[finite_rows].mean().item()
        )
    else:
        scalars[f"act/{name}/mean_per_sample_std"] = 0.0

    # Variability of each unit across the sampled batch (mean over units).
    if x.shape[0] > 1:
        counts_feat = finite.sum(dim=0)
        feat_mean = safe.sum(dim=0) / counts_feat.clamp_min(1)
        feat_centered = torch.where(finite, x - feat_mean[None, :], torch.zeros_like(x))
        feat_var = (feat_centered * feat_centered).sum(dim=0) / counts_feat.clamp_min(1)
        valid = counts_feat >= 2
        std_across = torch.sqrt(feat_var.clamp_min(0.0))[valid]
        scalars[f"act/{name}/std_across_samples"] = (
            float(std_across.mean().item()) if std_across.numel() else 0.0
        )
    else:
        scalars[f"act/{name}/std_across_samples"] = 0.0

    if saturation_abs is not None and finite_count:
        saturated = ((x.abs() >= saturation_abs) & finite).sum().item()
        scalars[f"act/{name}/saturated_fraction_abs_ge_{_saturation_threshold_tag(saturation_abs)}"] = (
            float(saturated / finite_count)
        )
    return scalars, per_sample


def _hook_targets(policy) -> List[Tuple[str, nn.Module]]:
    """Named modules to probe: visual encoders, projections, value MLP."""
    targets: List[Tuple[str, nn.Module]] = []
    extractor = getattr(policy, "features_extractor", None)
    if extractor is not None:
        for map_name in ("screen", "minimap"):
            encoder = getattr(extractor, f"_{map_name}_encoder", None)
            if isinstance(encoder, nn.Module):
                targets.append((f"features_{map_name}_encoder", encoder))
            fc = getattr(extractor, f"_{map_name}_fc", None)
            if isinstance(fc, nn.Module):
                seen = set()
                for module in fc:
                    if isinstance(module, nn.Linear) and "linear" not in seen:
                        targets.append((f"features_{map_name}_fc_linear", module))
                        seen.add("linear")
                    elif isinstance(module, nn.ReLU) and "relu" not in seen:
                        targets.append((f"features_{map_name}_fc_relu_pre_norm", module))
                        seen.add("relu")
                    elif isinstance(module, nn.LayerNorm) and "norm" not in seen:
                        targets.append((f"features_{map_name}_fc_norm", module))
                        seen.add("norm")

    mlp = getattr(policy, "mlp_extractor", None)
    value_net = getattr(mlp, "value_net", None)
    if isinstance(value_net, nn.Module):
        linear_ordinal = 0
        tanh_ordinal = 0
        for module in value_net:
            if isinstance(module, nn.Linear):
                targets.append((f"vf_linear_{linear_ordinal}", module))
                linear_ordinal += 1
            elif isinstance(module, nn.Tanh):
                targets.append((f"vf_tanh_{tanh_ordinal}", module))
                tanh_ordinal += 1
    return targets


def probe_value_network(
    policy,
    sample: RolloutSample,
    device: torch.device,
    saturation_abs: float = CRITIC_TANH_SATURATION_ABS,
) -> Optional[ProbeResult]:
    """One no_grad forward over the sample, with temporary activation hooks.

    Runs ``extract_features -> mlp_extractor.forward_critic -> value_net``,
    the same path ``predict_values`` uses. Hooks and the extractor's spatial
    map cache are always released in ``finally``.
    """
    mlp = getattr(policy, "mlp_extractor", None)
    value_head = getattr(policy, "value_net", None)
    if mlp is None or value_head is None:
        return None
    if not hasattr(policy, "extract_features"):
        return None

    batch = {
        key: torch.as_tensor(np.array(values, copy=True), device=device)
        for key, values in sample.observations.items()
    }
    if not batch:
        return None
    count = min(int(t.shape[0]) for t in batch.values())
    if count == 0:
        return None

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def _capture(name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach()
            elif isinstance(output, (tuple, list)) and output:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    captured[name] = first.detach()
        return _hook

    for name, module in _hook_targets(policy):
        handles.append(module.register_forward_hook(_capture(name)))

    tanh_names: List[str] = []
    try:
        with torch.no_grad():
            features = policy.extract_features(batch)
            latent_vf = mlp.forward_critic(features)
            values = value_head(latent_vf).reshape(-1)
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(policy, "_clear_last_spatial_maps"):
            policy._clear_last_spatial_maps()

    scalars: Dict[str, float] = {}
    per_sample: Dict[str, Dict[str, np.ndarray]] = {}
    tanh_saturation: List[float] = []
    for name, tensor in captured.items():
        is_tanh = name.startswith("vf_tanh_")
        stats, arrays = _module_stats(
            name, tensor, saturation_abs=saturation_abs if is_tanh else None
        )
        scalars.update(stats)
        per_sample.update(arrays)
        if is_tanh:
            tanh_names.append(name)
            fraction = stats.get(
                f"act/{name}/saturated_fraction_abs_ge_{_saturation_threshold_tag(saturation_abs)}"
            )
            if fraction is not None:
                tanh_saturation.append(float(fraction))

    # Final scalar value output statistics (saturation = outside the initial
    # Tanh range / very large values), separate from the vf latent layers.
    value_stats, _ = _module_stats("vf_output", values)
    scalars.update(value_stats)
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel():
        abs_values = finite_values.abs()
        scalars["act/vf_output/abs_ge_1_fraction"] = float(
            (abs_values >= CRITIC_VALUE_ABS_GE_1).float().mean().item()
        )
        scalars["act/vf_output/abs_ge_10_fraction"] = float(
            (abs_values >= CRITIC_VALUE_ABS_GE_10).float().mean().item()
        )
    if tanh_names:
        # Earlier diagnostics emitted this tag only when a Tanh value layer
        # existed; keep that contract so "no vf_tanh tags" stays meaningful.
        scalars["vf_tanh_saturation_sample_count"] = float(count)

    return ProbeResult(
        sample_count=count,
        values=values.detach().cpu().numpy().astype(np.float32),
        scalars=scalars,
        per_sample=per_sample,
        tanh_saturation=tanh_saturation,
    )


def _value_param_groups(policy) -> Dict[str, List[torch.nn.Parameter]]:
    """Small, named, disjoint parameter groups for the value-only gradient."""
    groups: Dict[str, List[torch.nn.Parameter]] = {}

    def _add(name: str, module) -> None:
        if not isinstance(module, nn.Module):
            return
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups[name] = params

    mlp = getattr(policy, "mlp_extractor", None)
    value_net = getattr(mlp, "value_net", None)
    if isinstance(value_net, nn.Sequential) and len(value_net) > 0:
        _add("vf_first_layer", value_net[0])
    _add("vf_last_head", getattr(policy, "value_net", None))

    extractor = getattr(policy, "features_extractor", None)
    for map_name in ("screen", "minimap"):
        fc = getattr(extractor, f"_{map_name}_fc", None)
        if isinstance(fc, nn.Sequential):
            for module in fc:
                if isinstance(module, nn.Linear):
                    _add(f"{map_name}_projection", module)
                    break
        _add(f"{map_name}_encoder", getattr(extractor, f"_{map_name}_encoder", None))
    _add("vector_ln", getattr(extractor, "_vector_ln", None))
    return groups


def _optimizer_param_ids(policy) -> Optional[set]:
    optimizer = getattr(policy, "optimizer", None)
    if optimizer is None:
        return None
    ids = set()
    try:
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                ids.add(id(param))
    except Exception:
        return None
    return ids


def compute_value_gradient_probe(
    policy,
    sample: RolloutSample,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    """Sampled value-MSE gradient norms without touching training state.

    This is the pre-update *sampled value-only* gradient (MSE of the current
    value against the stored GAE returns) on the same <=32 batch. It is not
    the full PPO gradient and it is not taken after clipping. Uses
    ``torch.autograd.grad`` so no ``param.grad`` is written.
    """
    if sample.returns is None or sample.values is None:
        return None
    groups = _value_param_groups(policy)
    if not groups:
        return None
    flat_params = [param for params in groups.values() for param in params]
    if not flat_params:
        return None

    batch = {
        key: torch.as_tensor(np.array(values, copy=True), device=device)
        for key, values in sample.observations.items()
    }
    if not batch:
        return None

    grads: Tuple[Optional[torch.Tensor], ...] = tuple()
    loss_value: Optional[float] = None
    loss_finite = False
    try:
        with torch.enable_grad():
            features = policy.extract_features(batch)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            values = policy.value_net(latent_vf).reshape(-1)
            target = torch.as_tensor(
                np.asarray(sample.returns, dtype=np.float32),
                device=values.device,
                dtype=values.dtype,
            )
            loss = torch.mean((values - target) * (values - target))
            loss_finite = bool(torch.isfinite(loss).item())
            if loss_finite:
                loss_value = float(loss.detach().item())
            grads = torch.autograd.grad(
                loss,
                flat_params,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
    finally:
        if hasattr(policy, "_clear_last_spatial_maps"):
            policy._clear_last_spatial_maps()
        features = latent_vf = values = loss = None  # noqa: F841

    optimizer_ids = _optimizer_param_ids(policy)
    scalars: Dict[str, float] = {
        "sample_count": float(sample.indices_flat.size),
        "loss_is_finite": 1.0 if loss_finite else 0.0,
        "optimizer_present": 0.0 if optimizer_ids is None else 1.0,
    }
    if loss_value is not None:
        scalars["value_mse_target_returns"] = loss_value

    grad_norms: Dict[str, Optional[float]] = {}
    param_norms: Dict[str, float] = {}
    finite_flags: Dict[str, bool] = {}
    used_fractions: Dict[str, float] = {}
    in_optimizer: Dict[str, Optional[float]] = {}

    cursor = 0
    with torch.no_grad():
        for name, params in groups.items():
            group_grads = grads[cursor:cursor + len(params)]
            cursor += len(params)
            used = [g for g in group_grads if g is not None]
            finite = all(bool(torch.isfinite(g).all().item()) for g in used)
            finite_flags[name] = finite
            used_fractions[name] = float(len(used) / len(params)) if params else 0.0
            total_sq = 0.0
            if finite:
                for grad in used:
                    total_sq += float(torch.linalg.vector_norm(grad.detach()).item()) ** 2
            norm = math.sqrt(total_sq) if finite and math.isfinite(total_sq) else None
            grad_norms[name] = norm
            param_sq = 0.0
            for param in params:
                param_sq += float(torch.linalg.vector_norm(param.detach()).item()) ** 2
            param_norm = math.sqrt(param_sq) if math.isfinite(param_sq) else float("inf")
            param_norms[name] = param_norm
            if optimizer_ids is not None:
                in_optimizer[name] = float(
                    sum(id(p) in optimizer_ids for p in params) / len(params)
                )

    for name in groups:
        if grad_norms.get(name) is not None:
            scalars[f"{name}_grad_l2"] = float(grad_norms[name])
        scalars[f"{name}_grad_finite"] = 1.0 if finite_flags.get(name) else 0.0
        scalars[f"{name}_grad_params_used_fraction"] = used_fractions.get(name, 0.0)
        if math.isfinite(param_norms.get(name, float("inf"))):
            scalars[f"{name}_param_l2"] = float(param_norms[name])
        if in_optimizer.get(name) is not None:
            scalars[f"{name}_param_in_optimizer_fraction"] = float(in_optimizer[name])
    return {
        "scalars": scalars,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "finite": finite_flags,
        "used_fractions": used_fractions,
        "value_mse": loss_value,
    }


def write_sample_npz(
    directory: str,
    step: int,
    sample: RolloutSample,
    probe: Optional[ProbeResult],
    gradient: Optional[Dict[str, object]],
    map_name: str,
    opt_step: int,
    sample_interval: int,
    sample_size: int = CRITIC_SATURATION_SAMPLE_SIZE,
) -> Tuple[str, str]:
    """Write the compressed sample and metadata sidecar without overwriting."""
    os.makedirs(directory, exist_ok=True)
    base = f"critic_rollout_step{int(step):012d}"
    stem = base
    suffix = 1
    while os.path.exists(os.path.join(directory, stem + ".npz")) or os.path.exists(
        os.path.join(directory, stem + "_meta.json")
    ):
        stem = f"{base}_{suffix}"
        suffix += 1
    npz_path = os.path.join(directory, stem + ".npz")
    meta_path = os.path.join(directory, stem + "_meta.json")
    # numpy appends ".npz" unless the path already ends with it, so the
    # temporary file must keep the extension.
    tmp_npz = os.path.join(directory, f"{stem}.{os.getpid()}.tmp.npz")

    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(DIAG_SAMPLE_SCHEMA_VERSION, dtype=np.int64),
        "num_timesteps": np.asarray(int(step), dtype=np.int64),
        "opt_step": np.asarray(int(opt_step), dtype=np.int64),
        "buffer_num_steps": np.asarray(sample.num_steps, dtype=np.int64),
        "buffer_num_envs": np.asarray(sample.num_envs, dtype=np.int64),
        "sample_indices_flat": sample.indices_flat.astype(np.int64),
        "sample_indices_time": sample.indices_time.astype(np.int64),
        "sample_indices_env": sample.indices_env.astype(np.int64),
    }
    for key, values in sample.observations.items():
        arrays[f"obs_{key}"] = values
    for name, values in (
        ("stored_values", sample.values),
        ("stored_returns", sample.returns),
        ("stored_advantages", sample.advantages),
        ("stored_rewards", sample.rewards),
        ("episode_starts", sample.episode_starts),
    ):
        if values is not None:
            arrays[name] = np.asarray(values)
    if probe is not None:
        arrays["current_values"] = np.asarray(probe.values, dtype=np.float32)
        for name, values in probe.per_sample.items():
            arrays[f"act_{name}"] = np.asarray(values, dtype=np.float32)
    if gradient is not None:
        # Non-finite metrics stay in the npz as NaN rather than disappearing;
        # the JSON sidecar reports them as null plus a finite fraction.
        for name, value in gradient["grad_norms"].items():
            arrays[f"grad_l2_{name}"] = np.asarray(
                np.nan if value is None or not math.isfinite(value) else value,
                dtype=np.float32,
            )
        for name, value in gradient["param_norms"].items():
            arrays[f"param_l2_{name}"] = np.asarray(
                np.nan if not math.isfinite(value) else value, dtype=np.float32
            )
        for name, value in gradient["finite"].items():
            arrays[f"grad_finite_{name}"] = np.asarray(1.0 if value else 0.0, dtype=np.float32)
        for name, value in gradient["used_fractions"].items():
            arrays[f"grad_params_used_fraction_{name}"] = np.asarray(value, dtype=np.float32)
        if gradient.get("value_mse") is not None:
            arrays["grad_value_mse_target_returns"] = np.asarray(
                gradient["value_mse"], dtype=np.float32
            )

    try:
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, npz_path)
    except Exception:
        try:
            if os.path.exists(tmp_npz):
                os.remove(tmp_npz)
        except Exception:
            pass
        raise

    array_schema = {
        name: {"dtype": str(arr.dtype), "shape": [int(x) for x in arr.shape]}
        for name, arr in arrays.items()
    }
    gradient_meta: Dict[str, object] = {
        "loss": "mse(current_values, stored_returns)",
        "method": "torch.autograd.grad",
        "scope": "sampled_value_only_preupdate",
        "note": (
            "Pre-update sampled value-only gradient, not the full PPO "
            "gradient and not taken after clipping."
        ),
        "writes_param_grad": False,
        "optimizer_step": False,
        "available": bool(gradient is not None),
    }
    if gradient is not None:
        groups_meta = {}
        for name, grad_l2 in gradient["grad_norms"].items():
            param_l2 = gradient["param_norms"].get(name)
            groups_meta[name] = {
                "grad_l2": (
                    float(grad_l2)
                    if grad_l2 is not None and math.isfinite(grad_l2)
                    else None
                ),
                "param_l2": (
                    float(param_l2) if param_l2 is not None and math.isfinite(param_l2) else None
                ),
                "grad_finite_fraction": 1.0 if gradient["finite"].get(name) else 0.0,
                "params_used_fraction": float(gradient["used_fractions"].get(name, 0.0)),
            }
        gradient_meta["groups"] = groups_meta
        value_mse = gradient.get("value_mse")
        gradient_meta["value_mse_target_returns"] = (
            float(value_mse) if value_mse is not None and math.isfinite(value_mse) else None
        )
    metadata = {
        "schema_version": DIAG_SAMPLE_SCHEMA_VERSION,
        "kind": "critic_rollout_sample",
        "timing": "rollout_end_before_ppo_train_preupdate",
        "map": map_name,
        "num_timesteps": int(step),
        "opt_step": int(opt_step),
        "rollout": {
            "num_steps": int(sample.num_steps),
            "num_envs": int(sample.num_envs),
            "transitions": int(sample.transition_count),
            "buffer_generator_ready_at_read": bool(sample.generator_ready),
        },
        "sample": {
            "count": int(sample.indices_flat.size),
            "max_samples": int(sample_size),
            "rule": "evenly_spaced_flat_indices_round_linspace_first_to_last_inclusive",
            "index_order": "flat = time_index * num_envs + env_index (time-major C-order)",
            "covers_full_rollout": bool(sample.indices_flat.size == sample.transition_count),
        },
        "sampling_schedule": {
            "interval_env_steps": int(sample_interval),
            "first_complete_rollout_saved": True,
            "num_timesteps_is_x_axis": True,
        },
        "thresholds": {
            "tanh_saturation_abs": float(CRITIC_TANH_SATURATION_ABS),
            "value_abs_ge_1": float(CRITIC_VALUE_ABS_GE_1),
            "value_abs_ge_10": float(CRITIC_VALUE_ABS_GE_10),
            "zero_variance_eps": float(ZERO_VARIANCE_EPS),
        },
        "gradient": gradient_meta,
        "npz_file": os.path.basename(npz_path),
        "arrays": array_schema,
        "created_wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if probe is None:
        metadata["probe_note"] = "activation probe failed; current_values absent"
    try:
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
    except Exception:
        logger.warning("critic 诊断: metadata 写入失败: %s", meta_path, exc_info=True)
    return npz_path, meta_path


class CriticDiagnostics:
    """Stateful orchestrator called from ``TBDualWriterCallback``."""

    def __init__(
        self,
        base_log_dir: str,
        sample_size: int = CRITIC_SATURATION_SAMPLE_SIZE,
        sample_interval_env_steps: int = DIAG_SAMPLE_INTERVAL_ENV_STEPS,
        diagnostics_dirname: str = DIAGNOSTICS_DIRNAME,
    ):
        self.base_log_dir = str(base_log_dir)
        self.sample_size = max(1, int(sample_size))
        self.sample_interval_env_steps = max(1, int(sample_interval_env_steps))
        self.diagnostics_dir = os.path.join(self.base_log_dir, diagnostics_dirname)
        self.last_sample_step: Optional[int] = None
        self.last_written_paths: Optional[Tuple[str, str]] = None

    def sample_due(self, step: int) -> bool:
        if self.last_sample_step is None:
            return True
        return int(step) - int(self.last_sample_step) >= self.sample_interval_env_steps

    def _build_probe(self, model):
        """Sample the rollout and run the single no_grad activation probe."""
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None:
            return None, None
        policy = getattr(model, "policy", None)
        if policy is None:
            return None, None
        try:
            sample = sample_rollout(buffer, self.sample_size)
        except Exception:
            logger.warning("critic 诊断: 采样 rollout 失败", exc_info=True)
            return None, None
        if sample is None:
            return None, None
        device = torch.device(getattr(model, "device", "cpu"))
        try:
            probe = probe_value_network(policy, sample, device)
        except Exception:
            logger.warning("critic 诊断: 激活探针失败，跳过本 rollout", exc_info=True)
            probe = None
        return sample, probe

    def _emit_scalars(self, buffer, probe, emit: Callable[[str, float], None]) -> None:
        try:
            for tag, value in compute_rollout_stats(buffer).items():
                emit(f"{CRITIC_DIAG_PREFIX}/{tag}", value)
        except Exception:
            logger.warning("critic 诊断: rollout 统计失败", exc_info=True)

        if probe is None:
            return
        for tag, value in probe.scalars.items():
            emit(f"{CRITIC_DIAG_PREFIX}/{tag}", value)
        # Layer ordinal among Tanh modules, not the Sequential module index.
        threshold = _saturation_threshold_tag(CRITIC_TANH_SATURATION_ABS)
        for layer_index, fraction in enumerate(probe.tanh_saturation):
            emit(
                f"{CRITIC_DIAG_PREFIX}/vf_tanh_layer{layer_index}_saturated_fraction_sample"
                f"{int(CRITIC_SATURATION_SAMPLE_SIZE)}_abs_ge_{threshold}",
                fraction,
            )

    def on_rollout_end(
        self,
        model,
        step: int,
        write_scalars: bool,
        emit: Callable[[str, float], None],
        map_name: Optional[str] = None,
        opt_step: int = 0,
    ) -> None:
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None:
            return
        if bool(getattr(buffer, "generator_ready", False)):
            logger.warning(
                "critic 诊断: rollout buffer 已被 get() 按 env-major 展平，跳过本次诊断"
            )
            return
        step = int(step)
        due = self.sample_due(step)
        if not write_scalars and not due:
            return

        # One no_grad probe per rollout, shared by scalars and the saved sample.
        sample, probe = self._build_probe(model)

        if write_scalars:
            self._emit_scalars(buffer, probe, emit)

        if not due or sample is None:
            return

        policy = getattr(model, "policy", None)
        if policy is None:
            return
        device = torch.device(getattr(model, "device", "cpu"))

        gradient: Optional[Dict[str, object]] = None
        try:
            gradient = compute_value_gradient_probe(policy, sample, device)
        except Exception:
            logger.warning("critic 诊断: 梯度探针失败，样本中梯度字段缺失", exc_info=True)
        if gradient is not None:
            for tag, value in gradient["scalars"].items():
                emit(f"{_GRAD_PREFIX}/{tag}", value)

        try:
            paths = write_sample_npz(
                self.diagnostics_dir,
                step=step,
                sample=sample,
                probe=probe,
                gradient=gradient,
                map_name=str(map_name) if map_name else "unknown",
                opt_step=opt_step,
                sample_interval=self.sample_interval_env_steps,
                sample_size=self.sample_size,
            )
        except Exception:
            logger.warning(
                "critic 诊断: 样本写入失败（不影响训练），下一步继续重试", exc_info=True
            )
            return
        self.last_sample_step = step
        self.last_written_paths = paths
