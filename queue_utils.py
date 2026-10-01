import redis
from rq import Queue

from config import Config

_redis_conn = None
_queue = None


def get_redis():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis.from_url(Config.REDIS_URL)
    return _redis_conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(Config.RQ_QUEUE_NAME, connection=get_redis())
    return _queue
