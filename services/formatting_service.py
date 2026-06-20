import json
from pathlib import Path
from typing import Any

from config import PRICE_SYMBOL


def _indian_grouping(amount: int) -> str:
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) <= 3:
        return sign + digits
    last_three = digits[-3:]
    leading = digits[:-3]
    groups = []
    while leading:
        groups.append(leading[-2:])
        leading = leading[:-2]
    return sign + ",".join(reversed(groups)) + "," + last_three


def format_currency(value: float) -> str:
    if value is None:
        return ""
    amount = int(round(value))
    return f"{PRICE_SYMBOL}{_indian_grouping(amount)}"


def format_number(value: float) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:,.2f}"


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).title()
