"""Generate the PySC2 action-parameter slot mapping reference table.

Usage:
    uv run python tools/generate_action_slot_map.py [OUTPUT_PATH]

The default output is ``docs/reference/action-slot-map.txt``. Importing this
module has no side effects; the file is only written by ``main()``.
"""
import argparse
from pathlib import Path

from pysc2.lib import actions

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "reference" / "action-slot-map.txt"


def build_slot_map():
    """Collect unique parameter slots and map every function id to them."""
    param_semantics = []  # 语义唯一的参数名列表
    param_semantics_set = set()
    param_size_dict = {}  # 语义->最大size
    for fn in actions.FUNCTIONS:
        for spec in fn.args:
            name = getattr(spec, "name")
            if name not in param_semantics_set:
                param_semantics.append(name)
                param_semantics_set.add(name)
            size = 1
            if getattr(spec, "sizes", None):
                if any(k in name for k in ["screen", "screen2"]):
                    size = 84 * 84
                elif "minimap" in name:
                    size = 64 * 64
                else:
                    size = int(spec.sizes[0])
            if name not in param_size_dict or size > param_size_dict[name]:
                param_size_dict[name] = size

    fn_param_map = {}
    for fn in actions.FUNCTIONS:
        slot_indices = []
        for spec in fn.args:
            name = getattr(spec, "name")
            slot_indices.append(param_semantics.index(name))
        fn_param_map[fn.id] = slot_indices

    return param_semantics, param_size_dict, fn_param_map


def render_slot_map(param_semantics, param_size_dict, fn_param_map):
    lines = ["[动作参数槽位分配表]"]
    for i, name in enumerate(param_semantics):
        lines.append(f"槽位{i}: {name}, size={param_size_dict[name]}")
    lines.append("")
    lines.append("[动作类型到槽位映射]")
    for fn in actions.FUNCTIONS:
        lines.append(
            f"fn_id={fn.id}, name={fn.name}, 槽位索引={fn_param_map[fn.id]}"
        )
    return "\n".join(lines) + "\n"


def write_slot_map(output_path=DEFAULT_OUTPUT):
    output_path = Path(output_path)
    param_semantics, param_size_dict, fn_param_map = build_slot_map()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_slot_map(param_semantics, param_size_dict, fn_param_map),
        encoding="utf-8",
    )
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成 PySC2 动作参数槽位映射表")
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help="输出文件路径（默认 docs/reference/action-slot-map.txt）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = write_slot_map(args.output)
    print(f"已输出到 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
