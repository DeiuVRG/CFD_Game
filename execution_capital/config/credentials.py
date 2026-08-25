import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from config.settings import BROKER


@dataclass
class Credentials:
    api_key: str
    identifier: str
    password: str
    mode: str
    base_url: str


def load_credentials() -> Credentials:
    load_dotenv()

    api_key = os.getenv("CAPITAL_API_KEY", "")
    identifier = os.getenv("CAPITAL_IDENTIFIER", "")
    password = os.getenv("CAPITAL_PASSWORD", "")
    mode = os.getenv("CAPITAL_MODE", "demo").lower()

    missing = []
    if not api_key:
        missing.append("CAPITAL_API_KEY")
    if not identifier:
        missing.append("CAPITAL_IDENTIFIER")
    if not password:
        missing.append("CAPITAL_PASSWORD")

    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your Capital.com credentials.")
        sys.exit(1)

    base_url = BROKER.LIVE_URL if mode == "live" else BROKER.DEMO_URL

    return Credentials(
        api_key=api_key,
        identifier=identifier,
        password=password,
        mode=mode,
        base_url=base_url,
    )
