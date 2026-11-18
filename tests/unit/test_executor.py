"""The worker pool must be usable after a server lifespan restarts."""
from app.api.routes import get_executor, shutdown_executor


def test_worker_pool_is_recreated_after_shutdown():
    first = get_executor()
    assert first.submit(lambda: 1).result(timeout=2) == 1
    shutdown_executor()
    second = get_executor()
    try:
        assert second is not first
        assert second.submit(lambda: 2).result(timeout=2) == 2
    finally:
        shutdown_executor()
