"""
Run indexing jobs pulled off Redis. Start with:
    python worker.py
or, with more than one process for throughput:
    rq worker indexing --url redis://localhost:6379/0
"""
import logging

from rq import Worker

import store
from config import Config
from queue_utils import get_redis

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

if __name__ == "__main__":
    store.init_db()
    problems = Config.validate()
    for p in problems:
        logging.warning(f"Config warning: {p}")

    logging.info(f"Starting RQ worker on queue '{Config.RQ_QUEUE_NAME}'...")
    worker = Worker([Config.RQ_QUEUE_NAME], connection=get_redis())
    worker.work()
