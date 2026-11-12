"""Execution policy for exact V2 normal neighbourhood queries, not sampling."""

import os


def normal_query_workers():
    value = int(os.environ.get("SHIP_REVIEW_NORMAL_WORKERS", "1"))
    if value < 1 or value > 4:
        raise ValueError("SHIP_REVIEW_NORMAL_WORKERS must be between 1 and 4")
    return value


class ExactQueryExecution:
    def __init__(self, tree, workers):
        self.tree = tree
        self.workers = workers

    def query(self, *args, **kwargs):
        kwargs["workers"] = self.workers
        return self.tree.query(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tree, name)
