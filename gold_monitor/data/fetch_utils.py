"""Fetch robustness helpers: retry with exponential backoff and optional
custom TLS session for yfinance (needed behind some corporate/agent proxies).

Environment knobs (all optional - default behavior is plain yfinance):
  YF_IMPERSONATE   curl_cffi TLS impersonation profile ('safari', 'chrome',
                   ...). Some TLS-terminating proxies reset the default
                   chrome fingerprint; 'safari' usually passes.
  HTTPS_PROXY      standard proxy env var, forwarded to the session.
  CURL_CA_BUNDLE / SSL_CERT_FILE
                   CA bundle for the proxy's re-terminated TLS.
"""
import logging
import os
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0


def retry_call(fn: Callable[[], T], retries: int = DEFAULT_RETRIES,
               base_delay: float = DEFAULT_BASE_DELAY,
               what: str = "fetch") -> T:
    """Call fn(); on exception retry with exponential backoff (2s, 4s, 8s...).
    Raises the last exception when all attempts fail."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network layer, retry anything
            last_exc = e
            if attempt < retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"{what} failed (attempt {attempt + 1}/{retries + 1}): {e}"
                    f" - retrying in {delay:.0f}s"
                )
                time.sleep(delay)
    raise last_exc


_session = None
_session_built = False


def get_yf_session():
    """Optional curl_cffi session for yfinance, controlled by YF_IMPERSONATE.
    Returns None (yfinance default) when the env var is unset or curl_cffi
    is unavailable. Cached after the first build."""
    global _session, _session_built
    if _session_built:
        return _session
    _session_built = True

    profile = os.getenv("YF_IMPERSONATE", "").strip()
    if not profile:
        return None

    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        logger.warning("YF_IMPERSONATE set but curl_cffi is not installed")
        return None

    kwargs = {"impersonate": profile}

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if proxy:
        kwargs["proxies"] = {"https": proxy, "http": proxy}

    ca_bundle = os.getenv("CURL_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if ca_bundle and os.path.exists(ca_bundle):
        kwargs["verify"] = ca_bundle

    _session = curl_requests.Session(**kwargs)
    logger.info(f"yfinance session: impersonate={profile}, proxy={'yes' if proxy else 'no'}")
    return _session
