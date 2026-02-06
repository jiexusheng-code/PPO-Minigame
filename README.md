RL Minigame PPO
================

Reward Mode
-----------

You can switch reward mode via env_kwargs in the config.

Example:

```
env_kwargs:
	reward_mode: custom
```

Supported modes:
- native: use PySC2 built-in reward (default)
- custom: use per-map custom reward (currently MoveToBeacon)
