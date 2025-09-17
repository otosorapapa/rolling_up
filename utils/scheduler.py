"""Lightweight scheduler for periodic connector synchronisation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List

import pandas as pd

from connectors.base import BaseConnector


Callback = Callable[[pd.DataFrame], None]


@dataclass
class ScheduledJob:
    connector: BaseConnector
    interval: timedelta
    callback: Callback
    next_run: datetime

    def should_run(self, now: datetime) -> bool:
        return now >= self.next_run

    def run(self) -> pd.DataFrame:
        data = self.connector.fetch_sales()
        self.callback(data)
        self.next_run = datetime.now() + self.interval
        return data


class DataSyncScheduler:
    """Minimal synchronous scheduler similar to ``schedule`` library."""

    def __init__(self):
        self.jobs: List[ScheduledJob] = []

    def schedule(self, connector: BaseConnector, minutes: int, callback: Callback) -> ScheduledJob:
        job = ScheduledJob(
            connector=connector,
            interval=timedelta(minutes=minutes),
            callback=callback,
            next_run=datetime.now(),
        )
        self.jobs.append(job)
        return job

    def run_pending(self) -> None:
        now = datetime.now()
        for job in self.jobs:
            if job.should_run(now):
                job.run()

    def cancel_all(self) -> None:
        self.jobs.clear()
