"""Application-local time helpers. Defaults to Asia/Kolkata for the hackathon demo."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Kolkata")


def current_date():
    try:
        return datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    except Exception:
        return datetime.now().date()


def current_datetime():
    try:
        return datetime.now(ZoneInfo(APP_TIMEZONE))
    except Exception:
        return datetime.now()
