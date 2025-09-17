"""Alert handling utilities including notification helpers."""

from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yaml


def load_alert_settings(path: str) -> Dict[str, Any]:
    """Load alert definition from a YAML file."""

    file_path = Path(path)
    if not file_path.exists():
        return {
            "thresholds": {
                "yoy": -0.1,
                "delta": -300000,
                "slope": -1.0,
            },
            "notification": {},
        }
    with open(file_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def save_alert_settings(path: str, settings: Dict[str, Any]) -> None:
    """Persist alert configuration back to YAML."""

    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(settings, fp, allow_unicode=True)


def build_alert_message(alerts: pd.DataFrame) -> str:
    """Create a plain-text alert summary."""

    if alerts is None or alerts.empty:
        return "現在の閾値を超えるアラートはありません。"
    rows = []
    for _, row in alerts.iterrows():
        rows.append(
            f"{row.get('product_name', row.get('product_code'))} | {row.get('metric')} = {row.get('actual')}"
        )
    return "\n".join(rows)


def send_alert(
    alerts: pd.DataFrame,
    settings: Dict[str, Any],
    *,
    email_config: Optional[Dict[str, Any]] = None,
    slack_webhook: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Send alerts via e-mail and Slack when configured."""

    message = build_alert_message(alerts)
    result = {"email_sent": False, "slack_sent": False, "message": message}

    if dry_run:
        return result

    if email_config:
        try:
            msg = EmailMessage()
            msg["Subject"] = email_config.get("subject", "売上アラート通知")
            msg["From"] = email_config["from"]
            msg["To"] = email_config["to"]
            msg.set_content(message)

            with smtplib.SMTP(email_config.get("host", "localhost"), email_config.get("port", 25)) as smtp:
                if email_config.get("use_tls"):
                    smtp.starttls()
                if email_config.get("username") and email_config.get("password"):
                    smtp.login(email_config["username"], email_config["password"])
                smtp.send_message(msg)
            result["email_sent"] = True
        except Exception as exc:  # pragma: no cover - network operation
            result["email_error"] = str(exc)

    if slack_webhook:
        try:
            payload = {"text": message}
            response = requests.post(slack_webhook, data=json.dumps(payload), timeout=10)
            response.raise_for_status()
            result["slack_sent"] = True
        except Exception as exc:  # pragma: no cover
            result["slack_error"] = str(exc)

    return result
