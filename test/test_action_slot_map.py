from pysc2.lib import actions

# 自动收集所有参数语义（唯一槽位）
param_semantics = []  # 语义唯一的参数名列表
param_semantics_set = set()
param_size_dict = {}  # 语义->最大size
for fn in actions.FUNCTIONS:
    for spec in fn.args:
        name = getattr(spec, "name", "")
        if name not in param_semantics_set:
            param_semantics.append(name)
            param_semantics_set.add(name)
        if "screen" in name or "minimap" in name:
            size = 64 * 64  # 默认64，实际可根据环境配置
        else:
            size = int(spec.sizes[0]) if getattr(spec, "sizes", None) else 1
        if name not in param_size_dict or size > param_size_dict[name]:
            param_size_dict[name] = size

# 构建fn_id到槽位索引映射
fn_param_map = {}
for fn in actions.FUNCTIONS:
    slot_indices = []
    for spec in fn.args:
        name = getattr(spec, "name", "")
        slot_indices.append(param_semantics.index(name))
    fn_param_map[fn.id] = slot_indices

with open("action_slot_map.txt", "w", encoding="utf-8") as f:
    f.write("[动作参数槽位分配表]\n")
    for i, name in enumerate(param_semantics):
        f.write(f"槽位{i}: {name}, size={param_size_dict[name]}\n")
    f.write("\n[动作类型到槽位映射]\n")
    for fn in actions.FUNCTIONS:
        f.write(f"fn_id={fn.id}, name={fn.name}, 槽位索引={fn_param_map[fn.id]}\n")
print("已输出到 action_slot_map.txt")
