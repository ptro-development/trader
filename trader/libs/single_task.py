from celery import Task

import redis

REDIS_CLIENT = redis.Redis()


def only_one(function=None, key="", timeout=None):
    """Enforce only one celery task at a time."""
    def _dec(run_func):
        """Decorator."""
        def _caller(*args, **kwargs):
            """Caller."""
            ret_value = None
            have_lock = False
            lock = REDIS_CLIENT.lock(key, timeout=timeout)
            try:
                have_lock = lock.acquire(blocking=False)
                if have_lock:
                    ret_value = run_func(*args, **kwargs)
            finally:
                if have_lock:
                    lock.release()
            return ret_value
        return _caller
    return _dec(function) if function is not None else _dec


class SingleTask(Task):
    """A task."""

    @only_one(key="SingleTask", timeout=60 * 5)
    def run(self, **kwargs):
        """Run task."""
        print("Acquired lock for up to 5 minutes and ran task!")
