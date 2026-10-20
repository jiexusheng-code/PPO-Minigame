"""Regression test: eval_every must not be scaled by n_envs.

SB3's EvalCallback.on_step is called once per VecEnv step, and one
optimization-step contains n_steps VecEnv steps (independent of n_envs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import compute_eval_freq

EVAL_EVERY = 10
N_STEPS = 16


def test_eval_freq_is_counted_in_vec_env_steps():
    assert compute_eval_freq(EVAL_EVERY, N_STEPS) == EVAL_EVERY * N_STEPS


def test_n_envs_does_not_amplify_eval_interval():
    for n_envs in (1, 4):
        eval_freq = compute_eval_freq(EVAL_EVERY, N_STEPS)
        # Each callback call advances all n_envs in parallel.
        env_steps_per_eval = eval_freq * n_envs
        opt_steps_per_eval = env_steps_per_eval / (N_STEPS * n_envs)
        assert opt_steps_per_eval == EVAL_EVERY


if __name__ == "__main__":
    test_eval_freq_is_counted_in_vec_env_steps()
    test_n_envs_does_not_amplify_eval_interval()
    print("test_eval_freq: OK")
