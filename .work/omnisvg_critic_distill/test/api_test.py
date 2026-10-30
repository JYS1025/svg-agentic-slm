"""Call GPT-5.6 Luna and check billing through BizRouter.

Install the client with ``pip install openai``, set the required key, then run:
    python test/bizrouter_gpt_5_6_luna.py "안녕하세요!"
    python test/bizrouter_gpt_5_6_luna.py --billing
"""

import json
import os
import sys
import ast
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


USAGE_BILLING_URL = "https://api.bizrouter.ai/usage/v1/billing"
DEFAULT_KEYS_FILE = Path(__file__).with_name("api_keys.txt")


def load_api_keys(keys_file: Path = DEFAULT_KEYS_FILE) -> dict[str, str]:
    """Load string API keys from a local KEY = \"value\" file without executing it."""
    keys: dict[str, str] = {}
    for line_number, raw_line in enumerate(keys_file.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid key declaration on line {line_number}: {raw_line!r}")
        name = name.strip()
        if not name.isidentifier():
            raise ValueError(f"Invalid key name on line {line_number}: {name!r}")
        parsed_value = ast.literal_eval(value.strip())
        if not isinstance(parsed_value, str):
            raise ValueError(f"API key {name!r} must be a string.")
        keys[name] = parsed_value
    return keys


def get_api_key(name: str) -> str:
    """Prefer an explicitly supplied environment variable, then use api_keys.txt."""
    value = os.environ.get(name)
    if value:
        return value
    value = load_api_keys().get(name)
    if value:
        return value
    raise KeyError(
        f"{name} is not set in the environment or {DEFAULT_KEYS_FILE}."
    )


def call_luna(api_key: str, prompt: str) -> str:
    """Return the model's text response using the supplied BizRouter API key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.bizrouter.ai/v1",
    )
    response = client.responses.create(
        model="openai/gpt-5.6-luna",
        input=prompt,
    )
    return response.output_text


def get_billing(usage_api_key: str) -> dict[str, Any]:
    """Return the organization's current BizRouter billing information."""
    request = Request(
        USAGE_BILLING_URL,
        headers={
            "Authorization": f"Bearer {usage_api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    if not isinstance(payload, dict) or not isinstance(payload.get("billing"), dict):
        raise ValueError("BizRouter billing response has an unexpected format.")
    return payload


def print_billing_summary(payload: dict[str, Any]) -> None:
    """Print the remaining balance and related billing values in KRW."""
    organization = payload.get("organization", {})
    billing = payload["billing"]

    def krw(field: str) -> str:
        value = billing.get(field)
        if value is None:
            return "없음"
        return f"{float(value):,.2f}원"

    print(f"조직: {organization.get('name', '알 수 없음')}")
    print(f"결제 방식: {billing.get('payment_type', '알 수 없음')}")
    print(f"사용 가능 잔액: {krw('available_credits')}")
    print(f"현재 보유 크레딧: {krw('current_credits')}")
    print(f"이번 달 사용액: {krw('current_period_usage_cost_krw')}")
    print(f"미청구 금액: {krw('outstanding_balance_krw')}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--billing"]:
        usage_api_key = get_api_key("BIZROUTER_USAGE_API_KEY")
        print_billing_summary(get_billing(usage_api_key))
    else:
        api_key = get_api_key("BIZROUTER_API_KEY")
        prompt = " ".join(sys.argv[1:]) or "안녕하세요!"
        print(call_luna(api_key, prompt))
