"""Contract tests: unified pipeline loop has no backend branching in loop body."""

import inspect

from cofai.engine import run_eval


def test_run_eval_exists():
    assert callable(run_eval.run_eval)


def test_run_eval_loop_body_no_backend_branching():
    src = inspect.getsource(run_eval.run_eval)
    # heuristic guardrails: these strings should not appear in loop body logic
    # (component build may still inspect cfg shape).
    assert "eval.backend" not in src
    assert "resolve_eval_backend" not in src
    assert "get_driver" not in src

