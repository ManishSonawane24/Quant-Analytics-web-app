from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Literal, Optional

import pandas as pd


Direction = Literal["above", "below"]


@dataclass
class AlertRule:
    """
    Simple user‑defined alert on a time‑series metric (e.g. z‑score).

    This is intentionally generic so you can extend it later to support
    other metrics (spread, correlation, volume, etc.).
    """

    id: str
    name: str
    symbol_y: str
    symbol_x: str
    metric: str  # e.g. "zscore"
    threshold: float
    direction: Direction = "above"
    active: bool = True


@dataclass
class TriggeredAlert:
    rule_id: str
    name: str
    timestamp: int
    value: float
    message: str


class AlertRegistry:
    """
    In‑memory registry of alert rules.

    For a production deployment you'd typically back this with Redis or
    a database so alerts persist across restarts and can be shared across
    multiple processes.
    """

    def __init__(self) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._lock = RLock()

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> None:
        with self._lock:
            self._rules.pop(rule_id, None)

    def list_rules(self) -> List[AlertRule]:
        with self._lock:
            return list(self._rules.values())

    def evaluate_zscore_alerts(
        self,
        symbol_y: str,
        symbol_x: str,
        zscore: pd.Series,
    ) -> List[TriggeredAlert]:
        """
        Evaluate all z‑score‑based alerts for the given pair.
        Returns a list of triggered alerts based on the last z value.
        """
        if zscore.empty:
            return []
        last_ts = int(zscore.index[-1].timestamp() * 1000)
        last_val = float(zscore.iloc[-1])
        triggers: List[TriggeredAlert] = []

        with self._lock:
            for rule in self._rules.values():
                if not rule.active:
                    continue
                if rule.metric != "zscore":
                    continue
                if rule.symbol_y.upper() != symbol_y.upper():
                    continue
                if rule.symbol_x.upper() != symbol_x.upper():
                    continue

                if rule.direction == "above" and last_val > rule.threshold:
                    msg = f"{rule.name}: z-score {last_val:.2f} > {rule.threshold}"
                elif rule.direction == "below" and last_val < rule.threshold:
                    msg = f"{rule.name}: z-score {last_val:.2f} < {rule.threshold}"
                else:
                    continue

                triggers.append(
                    TriggeredAlert(
                        rule_id=rule.id,
                        name=rule.name,
                        timestamp=last_ts,
                        value=last_val,
                        message=msg,
                    )
                )

        return triggers


# Global singleton registry used by the API layer.
alert_registry = AlertRegistry()



