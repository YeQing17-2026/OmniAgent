"""Best-effort LLM usage accounting.

The usage ledger is intentionally observational: failed or delayed accounting must
never fail an LLM response. Writes go through a bounded queue and a single
background SQLite writer so request concurrency is not limited by DB locks.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import urllib.request
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


_MODELS_DEV_CACHE_LOCK = threading.Lock()
_MODELS_DEV_CACHE: Dict[str, Dict[str, Any]] = {}
_PROVIDER_ALIASES = {"gemini": "google"}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _pricing_key(provider: str, model_id: str) -> str:
    return f"{provider}/{model_id}"


def _normalize_pricing_entry(
    cost: Dict[str, Any],
    source: str,
) -> Optional[Dict[str, Any]]:
    pricing = {
        "input": _as_float(cost.get("input")),
        "output": _as_float(cost.get("output")),
        "cache_read": _as_float(cost.get("cache_read")),
        "source": source,
    }
    if all(pricing[key] is None for key in ("input", "output", "cache_read")):
        return None
    return pricing


def load_models_dev_catalog(
    source_url: str,
    timeout_seconds: float = 2.0,
    cache_ttl_seconds: int = 900,
) -> Dict[str, Any]:
    """Fetch and cache the raw models.dev catalog."""
    if not source_url:
        return {}

    now = time.time()
    with _MODELS_DEV_CACHE_LOCK:
        cached = _MODELS_DEV_CACHE.get(source_url)
        if cached and cache_ttl_seconds > 0:
            if now - cached["fetched_at"] < cache_ttl_seconds:
                return cached["catalog"]

    try:
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "OmniAgent/usage-pricing"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except Exception as exc:
        logger.debug("models_dev_catalog_fetch_failed", url=source_url, error=str(exc))
        with _MODELS_DEV_CACHE_LOCK:
            cached = _MODELS_DEV_CACHE.get(source_url)
            return cached["catalog"] if cached else {}

    catalog = payload if isinstance(payload, dict) else {}
    with _MODELS_DEV_CACHE_LOCK:
        _MODELS_DEV_CACHE[source_url] = {"fetched_at": now, "catalog": catalog}
    return catalog


def load_models_dev_pricing_catalog(
    source_url: str,
    timeout_seconds: float = 2.0,
    cache_ttl_seconds: int = 900,
) -> Dict[str, Dict[str, Any]]:
    """Extract models.dev pricing data as USD per million tokens."""
    payload = load_models_dev_catalog(
        source_url=source_url,
        timeout_seconds=timeout_seconds,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    catalog: Dict[str, Dict[str, Any]] = {}
    model_catalog: Dict[str, Dict[str, Any]] = {}
    ambiguous_model_ids = set()
    price_keys = ("input", "output", "cache_read")
    for provider_id, provider_data in payload.items():
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            cost = model_data.get("cost")
            if not isinstance(cost, dict):
                continue
            pricing = _normalize_pricing_entry(cost, "models.dev")
            if pricing:
                model_key = str(model_id)
                catalog[_pricing_key(str(provider_id), model_key)] = pricing
                existing = model_catalog.get(model_key)
                if existing is None:
                    model_catalog[model_key] = pricing
                elif any(existing.get(key) != pricing.get(key) for key in price_keys):
                    ambiguous_model_ids.add(model_key)

    # Bare model_id fallback is intentionally conservative: if multiple
    # providers publish different prices for the same model_id, do not guess.
    # Users can still add an explicit provider/model override in config.
    for model_id, pricing in model_catalog.items():
        if model_id not in ambiguous_model_ids:
            catalog.setdefault(model_id, pricing)
    return catalog


def _lookup_pricing(
    provider: str,
    model_id: str,
    catalog: Dict[str, Dict[str, Any]],
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    provider = str(provider or "unknown")
    model_id = str(model_id or "unknown")
    aliased_provider = _PROVIDER_ALIASES.get(provider, provider)
    candidates = [
        _pricing_key(provider, model_id),
        _pricing_key(aliased_provider, model_id),
        model_id,
    ]

    merged: Optional[Dict[str, Any]] = None
    for key in candidates:
        if key in catalog:
            merged = dict(catalog[key])
            break

    if overrides:
        for key in candidates:
            override = overrides.get(key)
            if isinstance(override, dict):
                normalized = _normalize_pricing_entry(override, "config")
                if normalized:
                    merged = dict(merged or {})
                    for price_key in ("input", "output", "cache_read"):
                        if normalized.get(price_key) is not None:
                            merged[price_key] = normalized[price_key]
                    merged["source"] = "config"
                break

    return merged


def _cost_component(tokens: int, rate_per_million: Optional[float]) -> float:
    if rate_per_million is None or tokens <= 0:
        return 0.0
    return tokens * rate_per_million / 1_000_000


def enrich_usage_summary_with_pricing(
    summary: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Add best-effort USD cost estimates to usage summary rows."""
    result = dict(summary)
    totals = dict(result.get("totals") or {})
    by_model = []
    total_cost = 0.0
    priced_models = 0

    for row_data in result.get("by_model", []):
        row = dict(row_data)
        pricing = _lookup_pricing(
            str(row.get("provider", "unknown")),
            str(row.get("model_id", "unknown")),
            catalog,
            overrides,
        )
        if pricing:
            input_rate = pricing.get("input")
            output_rate = pricing.get("output")
            cache_read_rate = pricing.get("cache_read")
            if cache_read_rate is None:
                cache_read_rate = input_rate

            input_billable_tokens = _as_int(row.get("input_tokens_uncached")) + _as_int(
                row.get("cache_creation_input_tokens")
            )
            input_cost = _cost_component(input_billable_tokens, input_rate)
            cache_read_cost = _cost_component(
                _as_int(row.get("cache_read_input_tokens")) or _as_int(row.get("cached_input_tokens")),
                cache_read_rate,
            )
            output_cost = _cost_component(_as_int(row.get("output_tokens")), output_rate)
            row_cost = input_cost + cache_read_cost + output_cost
            total_cost += row_cost
            priced_models += 1
            row["pricing"] = {
                "source": pricing.get("source"),
                "input_per_million_usd": input_rate,
                "output_per_million_usd": output_rate,
                "cache_read_per_million_usd": cache_read_rate,
            }
            row["estimated_cost_usd"] = round(row_cost, 8)
        else:
            row["pricing"] = None
            row["estimated_cost_usd"] = None
        by_model.append(row)

    totals["estimated_cost_usd"] = round(total_cost, 8)
    totals["estimated_cost_scope"] = "returned_models"
    totals["priced_models"] = priced_models
    totals["unpriced_models"] = len(by_model) - priced_models
    result["totals"] = totals
    result["by_model"] = by_model
    result["pricing"] = {
        "catalog_models": len(catalog),
        "overrides": len(overrides or {}),
        "unit": "USD per million tokens",
    }
    return result


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    created_at_unix REAL NOT NULL,
    session_id TEXT,
    user_id TEXT,
    channel_id TEXT,
    source_component TEXT,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    input_tokens_uncached INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    raw_usage_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_events_model_time
ON usage_events(provider, model_id, created_at_unix);

CREATE INDEX IF NOT EXISTS idx_usage_events_session
ON usage_events(session_id);
"""


def _as_int(value: Any) -> int:
    """Convert provider usage values to non-negative integers."""
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _nested_int(data: Dict[str, Any], parent: str, child: str) -> int:
    nested = data.get(parent)
    if isinstance(nested, dict):
        return _as_int(nested.get(child))
    return 0


def _local_today_start_unix() -> float:
    now = datetime.now().astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def normalize_usage(
    usage: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    default_provider: str = "unknown",
    default_model: str = "unknown",
) -> Dict[str, Any]:
    """Normalize provider usage shapes into ledger columns."""
    raw_usage = usage or {}
    response_meta = metadata or {}
    call_context = context or {}

    input_tokens = _as_int(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
    )
    output_tokens = _as_int(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
    )

    nested_cached = _nested_int(raw_usage, "prompt_tokens_details", "cached_tokens")
    cached_input_tokens = _as_int(
        raw_usage.get(
            "cached_input_tokens",
            raw_usage.get("cached_tokens", nested_cached),
        )
    )
    cache_read_input_tokens = _as_int(
        raw_usage.get("cache_read_input_tokens", cached_input_tokens)
    )
    cache_creation_input_tokens = _as_int(raw_usage.get("cache_creation_input_tokens"))

    explicit_uncached = raw_usage.get("input_tokens_uncached")
    if explicit_uncached is None:
        input_tokens_uncached = max(input_tokens - cached_input_tokens, 0)
    else:
        input_tokens_uncached = _as_int(explicit_uncached)

    total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = (
            input_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens
            + output_tokens
        )

    default_provider_value = default_provider if default_provider != "unknown" else None
    provider = str(
        call_context.get("provider")
        or default_provider_value
        or response_meta.get("provider")
        or "unknown"
    )
    model_id = str(
        call_context.get("model_id")
        or response_meta.get("model")
        or default_model
        or "unknown"
    )

    return {
        "session_id": call_context.get("session_id"),
        "user_id": call_context.get("user_id"),
        "channel_id": call_context.get("channel_id"),
        "source_component": call_context.get("source_component", "llm"),
        "provider": provider,
        "model_id": model_id,
        "input_tokens": input_tokens,
        "input_tokens_uncached": input_tokens_uncached,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_usage_json": json.dumps(raw_usage, ensure_ascii=False, sort_keys=True),
    }


class UsageRecorder:
    """SQLite-backed usage ledger with best-effort non-blocking writes."""

    def __init__(self, db_path: Path, max_queue_size: int = 10000):
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.dropped_events = 0
        self.failed_events = 0
        self._enabled = True
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._stop_event = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="omniagent-usage-writer",
            daemon=True,
        )

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema(self.db_path)
            self._writer.start()
        except Exception as exc:
            self._enabled = False
            logger.warning("usage_recorder_disabled", error=str(exc))

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=1000")

    @classmethod
    def _ensure_schema(cls, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        try:
            cls._configure_connection(conn)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def record(self, event: Dict[str, Any]) -> None:
        """Queue a usage event without blocking the caller."""
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped_events += 1
            logger.debug("usage_event_dropped", reason="queue_full")

    def record_response(
        self,
        response: Any,
        context: Optional[Dict[str, Any]] = None,
        default_provider: str = "unknown",
        default_model: str = "unknown",
    ) -> None:
        """Normalize and queue an LLM response usage event."""
        try:
            event = normalize_usage(
                response.usage,
                metadata=response.metadata,
                context=context,
                default_provider=default_provider,
                default_model=default_model,
            )
            if (
                event["total_tokens"] == 0
                and event["input_tokens"] == 0
                and event["output_tokens"] == 0
            ):
                return
            self.record(event)
        except Exception as exc:
            self.failed_events += 1
            logger.debug("usage_event_normalize_failed", error=str(exc))

    def _writer_loop(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=1.0)
            self._configure_connection(conn)
            while not self._stop_event.is_set():
                try:
                    event = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if event is None:
                    self._queue.task_done()
                    break

                try:
                    self._insert_event(conn, event)
                    conn.commit()
                except Exception as exc:
                    self.failed_events += 1
                    logger.debug("usage_event_write_failed", error=str(exc))
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._enabled = False
            logger.warning("usage_writer_stopped", error=str(exc))
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, event: Dict[str, Any]) -> None:
        now = time.time()
        created_at = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO usage_events (
                id, created_at, created_at_unix, session_id, user_id, channel_id,
                source_component, provider, model_id, input_tokens,
                input_tokens_uncached, cache_creation_input_tokens,
                cache_read_input_tokens, cached_input_tokens, output_tokens,
                total_tokens, raw_usage_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                created_at,
                now,
                event.get("session_id"),
                event.get("user_id"),
                event.get("channel_id"),
                event.get("source_component"),
                event.get("provider", "unknown"),
                event.get("model_id", "unknown"),
                event.get("input_tokens", 0),
                event.get("input_tokens_uncached", 0),
                event.get("cache_creation_input_tokens", 0),
                event.get("cache_read_input_tokens", 0),
                event.get("cached_input_tokens", 0),
                event.get("output_tokens", 0),
                event.get("total_tokens", 0),
                event.get("raw_usage_json"),
            ),
        )

    def summary(self, days: Optional[int] = None, limit: int = 50) -> Dict[str, Any]:
        """Return aggregate usage grouped by provider/model.

        When days is omitted, the summary defaults to the local calendar day
        starting at 00:00. Explicit days values keep their rolling-window
        meaning and are validated by API callers.
        """
        self._ensure_schema(self.db_path)
        safe_limit = max(1, min(limit, 500))
        where = "WHERE created_at_unix >= ?"
        if days is None:
            window_label = "today"
            params: List[Any] = [_local_today_start_unix()]
        else:
            window_label = f"last_{days}_days"
            params = [time.time() - days * 86400]

        conn = sqlite3.connect(str(self.db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            self._configure_connection(conn)
            totals = conn.execute(
                f"""SELECT
                    COUNT(*) AS events,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(input_tokens_uncached), 0) AS input_tokens_uncached,
                    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_events {where}""",
                params,
            ).fetchone()
            by_model = conn.execute(
                f"""SELECT
                    provider,
                    model_id,
                    COUNT(*) AS events,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(input_tokens_uncached), 0) AS input_tokens_uncached,
                    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_events {where}
                GROUP BY provider, model_id
                ORDER BY total_tokens DESC
                LIMIT ?""",
                [*params, safe_limit],
            ).fetchall()

            return {
                "window": window_label,
                "window_days": days,
                "queue_size": self._queue.qsize() if self._enabled else 0,
                "queue_capacity": self.max_queue_size,
                "dropped_events": self.dropped_events,
                "failed_events": self.failed_events,
                "totals": dict(totals) if totals else {},
                "by_model": [dict(row) for row in by_model],
            }
        finally:
            conn.close()

    def close(self) -> None:
        """Stop the background writer after queued events drain opportunistically."""
        if not self._enabled:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self.dropped_events += 1
            logger.debug("usage_close_signal_dropped", reason="queue_full")


class UsageTrackingLLMProvider:
    """LLMProvider proxy that records usage without blocking LLM responses."""

    def __init__(
        self,
        inner: Any,
        recorder: UsageRecorder,
        default_provider: str = "unknown",
        default_model: str = "unknown",
    ):
        self.inner = inner
        self.recorder = recorder
        self.default_provider = default_provider
        self.default_model = default_model
        self._context: ContextVar[Dict[str, Any]] = ContextVar(
            "omniagent_usage_context",
            default={},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def supports_native_function_calling(self) -> bool:
        return self.inner.supports_native_function_calling

    def set_usage_context(self, context: Dict[str, Any]) -> Token:
        return self._context.set(dict(context))

    def reset_usage_context(self, token: Token) -> None:
        self._context.reset(token)

    async def chat(
        self,
        messages: List[Any],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> Any:
        response = await self.inner.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
        )
        response.metadata["provider"] = self.default_provider
        response.metadata.setdefault("model", self.default_model)
        self.recorder.record_response(
            response,
            context=self._context.get(),
            default_provider=self.default_provider,
            default_model=self.default_model,
        )
        return response

    async def chat_stream(
        self,
        messages: List[Any],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        async for chunk in self.inner.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
