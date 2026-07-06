"""yfinance呼び出しの共通リトライ・バックオフ処理"""

import random
import time

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

MAX_ATTEMPTS = 3
CONSECUTIVE_FAILURE_LIMIT = 20  # これを超えたら早期打ち切り(ブロック検知)


class TooManyFailuresError(Exception):
    """短時間で連続失敗が続いた場合(Yahoo側のブロック疑い)に送出する"""


def jitter_sleep(base: float = 1.0, spread: float = 0.5) -> None:
    time.sleep(base + random.uniform(0, spread))


@retry(
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_info(ticker_symbol: str) -> dict:
    t = yf.Ticker(ticker_symbol)
    info = t.info
    if not info:
        raise ValueError(f"empty info for {ticker_symbol}")
    return info


@retry(
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_history(ticker_symbol: str, period: str = "5d"):
    t = yf.Ticker(ticker_symbol)
    df = t.history(period=period)
    return df


class ConsecutiveFailureGuard:
    """連続失敗数を数え、閾値を超えたらブロック疑いとして打ち切るためのガード"""

    def __init__(self, limit: int = CONSECUTIVE_FAILURE_LIMIT):
        self.limit = limit
        self.consecutive_failures = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.limit:
            raise TooManyFailuresError(
                f"{self.consecutive_failures}件連続で失敗しました。Yahoo側のブロックの可能性があるため処理を打ち切ります。"
            )
