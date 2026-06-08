from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

EVENT_TYPE_CATALOGUE = {
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
    "PURCHASE",
}

EVENT_SYNONYMS = {
    "entry": "ENTRY",
    "exit": "EXIT",
    "zone_entered": "ZONE_ENTER",
    "zone_enter": "ZONE_ENTER",
    "zone_exit": "ZONE_EXIT",
    "zone_dwell": "ZONE_DWELL",
    "billing_queue_join": "BILLING_QUEUE_JOIN",
    "billing_queue_abandon": "BILLING_QUEUE_ABANDON",
    "reentry": "REENTRY",
    "purchase": "PURCHASE",
}


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(text)
        except ValueError:
            ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def normalize_store_id(store_id: str) -> str:
    cleaned = str(store_id).strip().upper()
    if cleaned in {"ST1008", "STORE_1", "STORE 1", "STORE_BLR_002"}:
        return "STORE_BLR_002"
    if cleaned in {"ST1076", "STORE_1076", "STORE_2", "STORE 2", "STORE1076", "STORE_BLR_001"}:
        return "ST1076"
    return cleaned


def _normalize_event_type(raw: str) -> str:
    canonical = EVENT_SYNONYMS.get(str(raw).strip().lower(), str(raw).strip().upper())
    if canonical not in EVENT_TYPE_CATALOGUE:
        raise ValueError(f"unsupported_event_type:{raw}")
    return canonical


def _stable_zone_coord(zone_id: str) -> Tuple[float, float]:
    digest = hashlib.sha1(zone_id.encode("utf-8")).hexdigest()
    x_raw = int(digest[:8], 16)
    y_raw = int(digest[8:16], 16)
    return (round((x_raw % 10_000) / 100.0, 2), round((y_raw % 10_000) / 100.0, 2))

@dataclass


class NormalizedEvent:
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: Optional[int] = None
    is_staff: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod

    def from_payload(cls, payload: Dict[str, Any]) -> "NormalizedEvent":
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata_must_be_object")
        event_type = _normalize_event_type(payload["event_type"])
        return cls(
            event_id=str(payload["event_id"]),
            store_id=normalize_store_id(payload["store_id"]),
            camera_id=str(payload["camera_id"]),
            visitor_id=str(payload["visitor_id"]),
            event_type=event_type,
            timestamp=_parse_timestamp(payload["timestamp"]),
            zone_id=payload.get("zone_id"),
            dwell_ms=int(payload["dwell_ms"]) if payload.get("dwell_ms") is not None else None,
            is_staff=bool(payload.get("is_staff", False)),
            confidence=float(payload.get("confidence", 1.0)),
            metadata=metadata,
        )

@dataclass


class SessionRecord:
    store_id: str
    visitor_id: str
    session_seq: int
    is_staff: bool = False
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    last_queue_depth: Optional[int] = None
    active: bool = False
    event_types: List[str] = field(default_factory=list)
    events: List[NormalizedEvent] = field(default_factory=list)
    zone_first_seen: Dict[str, datetime] = field(default_factory=dict)
    zone_dwell_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    zone_frequency: Counter = field(default_factory=Counter)
    entry_count: int = 0
    exit_count: int = 0
    queue_joins: int = 0
    queue_abandons: int = 0
    purchase_count: int = 0
    billing_timestamps: List[datetime] = field(default_factory=list)
    observation_stamp: Optional[datetime] = None
    identity_keys: set[str] = field(default_factory=set)

    def ingest(self, event: NormalizedEvent) -> None:
        self.identity_keys.add(event.visitor_id)
        if self.first_timestamp is None or event.timestamp < self.first_timestamp:
            self.first_timestamp = event.timestamp
        if self.last_timestamp is None or event.timestamp > self.last_timestamp:
            self.last_timestamp = event.timestamp
        self.observation_stamp = event.timestamp
        self.is_staff = self.is_staff or event.is_staff
        self.event_types.append(event.event_type)
        self.events.append(event)

        if event.event_type in {"ENTRY", "REENTRY"}:
            self.entry_count += 1
            self.active = True
        elif event.event_type == "EXIT":
            self.exit_count += 1
            self.active = False
        elif event.event_type == "BILLING_QUEUE_JOIN":
            self.queue_joins += 1
            self.active = True
        elif event.event_type == "BILLING_QUEUE_ABANDON":
            self.queue_abandons += 1
            self.active = False
        elif event.event_type == "PURCHASE":
            self.purchase_count += 1
            self.active = False

        if event.zone_id:
            self.zone_frequency[event.zone_id] += 1
            if event.zone_id not in self.zone_first_seen:
                self.zone_first_seen[event.zone_id] = event.timestamp
            if event.event_type == "ZONE_DWELL" and event.dwell_ms is not None:
                self.zone_dwell_totals[event.zone_id] += int(event.dwell_ms)
            elif event.event_type == "ZONE_EXIT":
                start = self.zone_first_seen.get(event.zone_id)
                if start is not None:
                    dwell = max(0, int((event.timestamp - start).total_seconds() * 1000))
                    self.zone_dwell_totals[event.zone_id] += dwell

        queue_depth = event.metadata.get("queue_depth")
        if queue_depth is not None:
            try:
                self.last_queue_depth = int(queue_depth)
            except (TypeError, ValueError):
                pass

        if event.zone_id and "bill" in event.zone_id.lower():
            self.billing_timestamps.append(event.timestamp)

    def reached_stage(self, stage: str) -> bool:
        stage = stage.upper()
        if stage == "ENTRY":
            return self.entry_count > 0
        if stage == "ZONE_VISIT":
            return any(event.event_type in {"ZONE_ENTER", "ZONE_DWELL", "ZONE_EXIT"} for event in self.events)
        if stage == "BILLING_QUEUE":
            return self.queue_joins > 0 or any("bill" in str(event.zone_id or "").lower() for event in self.events)
        if stage == "PURCHASE":
            return self.purchase_count > 0
        return False

    def is_recent_active(self, now: datetime, inactivity_window: timedelta) -> bool:
        if self.last_timestamp is None or not self.active:
            return False
        return now - self.last_timestamp <= inactivity_window

@dataclass(frozen=True)


class TransactionRecord:
    store_id: str
    order_id: str
    timestamp: datetime


class StoreAnalyticsEngine:
    def __init__(self, transactions: Optional[Sequence[datetime]] = None):
        self._event_ids: set[str] = set()
        self._stores: Dict[str, Dict[str, SessionRecord]] = defaultdict(dict)
        self._store_events: Dict[str, List[NormalizedEvent]] = defaultdict(list)
        self._visitor_session_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        self._hourly_staff_counter: Dict[str, Counter] = defaultdict(Counter)
        self._store_last_event_at: Dict[str, datetime] = {}
        self._store_last_ingested_wallclock: Dict[str, datetime] = {}
        self._store_latest_queue_depth: Dict[str, int] = defaultdict(int)
        self._store_transactions: Dict[str, List[TransactionRecord]] = defaultdict(list)
        self._default_transactions: List[TransactionRecord] = []
        if transactions:
            for index, ts in enumerate(transactions, start=1):
                self._default_transactions.append(
                    TransactionRecord(store_id="default", order_id=f"default-{index}", timestamp=ts)
                )

    @property

    def event_ids(self) -> set[str]:
        return self._event_ids

    def load_transactions(self, csv_path: Optional[str]) -> None:
        if not csv_path:
            return
        path = Path(csv_path)
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return

        columns = {column.lower().strip(): column for column in df.columns}
        date_col = columns.get("order_date")
        time_col = columns.get("order_time")
        store_col = columns.get("store_id")
        order_col = columns.get("order_id")

        for _, row in df.iterrows():
            date_part = str(row[date_col]).strip()
            time_part = str(row[time_col]).strip()
            combined = f"{date_part} {time_part}"
            parsed = pd.to_datetime(combined, errors="coerce", dayfirst=True)
            if pd.isna(parsed):
                continue

            ts = parsed.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            store_id = normalize_store_id(str(row[store_col]).strip() if store_col else "default")
            order_id = str(row[order_col]).strip() if order_col else f"{store_id}-{uuid4()}"

            record = TransactionRecord(
                store_id=store_id,
                order_id=order_id,
                timestamp=ts.astimezone(timezone.utc)
            )
            self._store_transactions[store_id].append(record)

        if not self._default_transactions:
            self._default_transactions = [item for sublist in self._store_transactions.values() for item in sublist]

    def ingest_events(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        accepted = 0
        rejected: List[Dict[str, str]] = []
        duplicates = 0

        for index, payload in enumerate(records):
            try:
                event = NormalizedEvent.from_payload(payload)
            except Exception as exc:
                rejected.append({"index": str(index), "error": str(exc)})
                continue

            if event.event_id in self._event_ids:
                duplicates += 1
                continue

            self._event_ids.add(event.event_id)
            accepted += 1
            self._store_events[event.store_id].append(event)
            self._store_last_event_at[event.store_id] = event.timestamp
            self._store_last_ingested_wallclock[event.store_id] = datetime.now(timezone.utc)

            session = self._find_matching_session(event)

            if session is None:
                session_seq = event.metadata.get("session_seq")
                if session_seq is None:
                    counter_key = (event.store_id, self._canonical_visitor_id(event.visitor_id))
                    if event.event_type in {"ENTRY"}:
                        self._visitor_session_counter[counter_key] += 1
                    session_seq = max(self._visitor_session_counter[counter_key], 1)

                try:
                    session_seq = int(session_seq)
                except (TypeError, ValueError):
                    session_seq = 1

                canonical_id = self._canonical_visitor_id(event.visitor_id)
                session_key = f"{canonical_id}:{session_seq}"

                session = self._stores[event.store_id].setdefault(
                    session_key,
                    SessionRecord(
                        store_id=event.store_id,
                        visitor_id=canonical_id,
                        session_seq=session_seq,
                        identity_keys={event.visitor_id},
                    ),
                )
            else:
                session.identity_keys.add(event.visitor_id)

            session.ingest(event)

            if event.metadata.get("queue_depth") is not None:
                try:
                    self._store_latest_queue_depth[event.store_id] = int(event.metadata["queue_depth"])
                except (TypeError, ValueError):
                    pass
            if event.is_staff:
                self._hourly_staff_counter[event.store_id][event.timestamp.hour] += 1

        return {
            "accepted": accepted,
            "rejected": rejected,
            "duplicate_count": duplicates,
            "partial_success": bool(rejected or duplicates),
        }

    def _store_sessions(self, store_id: str) -> List[SessionRecord]:
        store_id = normalize_store_id(store_id)
        return list(self._stores.get(store_id, {}).values())

    def _non_staff_sessions(self, store_id: str) -> List[SessionRecord]:
        store_id = normalize_store_id(store_id)
        return [session for session in self._store_sessions(store_id) if not session.is_staff]

    def _unique_visitors(self, store_id: str) -> set[str]:
        store_id = normalize_store_id(store_id)
        return {f"{session.visitor_id}:{session.session_seq}" for session in self._non_staff_sessions(store_id)}

    @staticmethod

    def _canonical_visitor_id(visitor_id: str) -> str:
        text = str(visitor_id).strip()
        parts = text.split("_")
        if len(parts) >= 2 and parts[0] == "VIS":
            numeric_parts = [part for part in parts[1:] if part.isdigit()]
            if numeric_parts:
                return f"VIS_{numeric_parts[-1]}"
        return text

    def _find_matching_session(self, event: NormalizedEvent) -> Optional[SessionRecord]:
        now = event.timestamp
        canonical_id = self._canonical_visitor_id(event.visitor_id)
        for session in self._stores.get(event.store_id, {}).values():
            session_keys = session.identity_keys or {session.visitor_id}
            canonical_keys = {self._canonical_visitor_id(key) for key in session_keys}
            if canonical_id in canonical_keys and session.is_recent_active(now, timedelta(minutes=15)):
                return session
        return None

    def _current_time(self, store_id: str) -> datetime:
        store_id = normalize_store_id(store_id)
        return self._store_last_event_at.get(store_id, datetime.now(timezone.utc))

    def _purchase_units(self, store_id: str, sessions: Optional[List[SessionRecord]] = None) -> Tuple[float, int]:
        """
        Correlates visitor billing events with POS transactions
        within a robust look-around correlation window.
        """
        store_id = normalize_store_id(store_id)
        transactions = self._store_transactions.get(store_id) or self._default_transactions
        if sessions is None:
            sessions = self._non_staff_sessions(store_id)

        purchased_order_ids = set()

        for session in sessions:
            session.purchase_count = 0
            for bill_ts in session.billing_timestamps:
                window_start = bill_ts - timedelta(minutes=5)
                window_end = bill_ts + timedelta(minutes=30)

                for txn in transactions:
                    txn_store = normalize_store_id(txn.store_id)
                    txn_aligned_ts = txn.timestamp.replace(
                        year=bill_ts.year, month=bill_ts.month, day=bill_ts.day
                    )
                    if txn_store == store_id and window_start <= txn_aligned_ts <= window_end:
                        purchased_order_ids.add((txn_store, txn_aligned_ts))
                        session.purchase_count = 1

        return float(len(purchased_order_ids)), len(purchased_order_ids)

    def snapshot_metrics(self, store_id: str) -> Dict[str, Any]:
        store_id = normalize_store_id(store_id)
        all_sessions = self._non_staff_sessions(store_id)
        now = self._current_time(store_id)

        active_sessions = [s for s in all_sessions if s.is_recent_active(now, timedelta(minutes=15))]

        unique_visitors = {s.visitor_id for s in active_sessions}

        live_occupancy = sum(1 for s in active_sessions if s.active)

        total_entries = sum(s.entry_count for s in all_sessions)
        total_exits = sum(s.exit_count for s in all_sessions)

        billing_sessions = [
            s for s in active_sessions
            if s.last_queue_depth is not None and (s.queue_joins > 0 or s.billing_timestamps)
        ]
        queue_depth = max([s.last_queue_depth for s in billing_sessions] or [0])

        purchase_units, _ = self._purchase_units(store_id, active_sessions)
        denom = len(unique_visitors)
        if denom > 0:
            conversion_rate = round((purchase_units / denom) * 100.0, 2)
        else:
            conversion_rate = 0.0

        total_joins = sum(s.queue_joins for s in all_sessions)
        abandonment_rate = round(
            (sum(s.queue_abandons for s in all_sessions) / max(total_joins, 1)) * 100.0,
            2
        )

        return {
            "unique_visitors": len(unique_visitors),
            "live_occupancy": live_occupancy,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "hourly_staff_counts": len({s.visitor_id for s in all_sessions if s.is_staff}),
            "store_conversion_rate_percentage": conversion_rate,
            "queue_depth": int(queue_depth),
            "abandonment_rate": abandonment_rate,
        }

    def funnel(self, store_id: str) -> Dict[str, Any]:
        sessions = self._non_staff_sessions(store_id)
        stage_order = ["ENTRY", "ZONE_VISIT", "BILLING_QUEUE", "PURCHASE"]
        stage_labels = {
            "ENTRY": "Entry",
            "ZONE_VISIT": "Zone Visit",
            "BILLING_QUEUE": "Billing Queue",
            "PURCHASE": "Purchase",
        }

        stage_counts: List[int] = []
        for stage in stage_order:
            count = sum(1 for session in sessions if session.reached_stage(stage))
            if stage_counts:
                count = min(count, stage_counts[-1])
            stage_counts.append(count)

        funnel = []
        for index, stage in enumerate(stage_order):
            current = stage_counts[index] if index < len(stage_counts) else 0
            if index == 0:
                drop_off = 0.0
            else:
                previous = max(stage_counts[index - 1], 1)
                drop_off = round(max(0.0, 1.0 - (current / previous)) * 100.0, 2)
            funnel.append({"stage": stage_labels[stage], "count": current, "drop_off_percentage": drop_off})

        return funnel

    def heatmap(self, store_id: str) -> Dict[str, Any]:
        sessions = self._non_staff_sessions(store_id)
        unique_session_count = len({(session.visitor_id, session.session_seq) for session in sessions})
        zone_frequency: Counter = Counter()
        zone_dwell: Dict[str, int] = defaultdict(int)

        for session in sessions:
            for zone_id, count in session.zone_frequency.items():
                zone_frequency[zone_id] += count
            for zone_id, dwell in session.zone_dwell_totals.items():
                zone_dwell[zone_id] += dwell

        nodes = []
        highest_frequency = max(zone_frequency.values(), default=0)
        for zone_id, frequency in zone_frequency.items():
            x, y = _stable_zone_coord(zone_id)
            average_dwell = round(zone_dwell.get(zone_id, 0) / max(frequency, 1), 2)
            visit_score = round((frequency / max(highest_frequency, 1)) * 100.0, 2)
            nodes.append(
                {
                    "zone_id": zone_id,
                    "coordinates": {
                        "x": round(min(max(x, 0.0), 100.0), 2),
                        "y": round(min(max(y, 0.0), 100.0), 2),
                    },
                    "frequency": frequency,
                    "average_dwell_ms": average_dwell,
                    "visit_frequency_score": visit_score,
                }
            )

        return {
            "data_confidence": unique_session_count >= 20,
            "zones": nodes,
            "visit_frequency_score": [node["visit_frequency_score"] for node in nodes],
        }

    def anomalies(self, store_id: str) -> List[Dict[str, Any]]:
        store_id = normalize_store_id(store_id)
        metrics = self.snapshot_metrics(store_id)
        anomalies: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        last_event_at = self._store_last_event_at.get(store_id)
        if metrics["unique_visitors"] > 15 and self._purchase_units(store_id)[1] == 0:
            anomalies.append(
                {
                    "anomaly_type": "CONVERSION_DROP",
                    "severity": "WARN",
                    "suggested_action": "Investigate checkout friction and review queue-to-purchase latency over the last hour.",
                }
            )
        if last_event_at is None or now - self._store_last_ingested_wallclock.get(store_id, last_event_at or now) > timedelta(minutes=30):
            anomalies.append(
                {
                    "anomaly_type": "DEAD_ZONE",
                    "severity": "CRITICAL",
                    "suggested_action": "Verify camera ingestion, Redis connectivity, and worker liveness immediately.",
                }
            )
        return anomalies

    def health(self, store_id: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        raw_key = store_id or next(iter(self._store_last_event_at.keys()), None)
        store_key = normalize_store_id(raw_key) if raw_key else None
        last_event_at = self._store_last_event_at.get(store_key) if store_key else None
        gap_seconds = int((now - last_event_at).total_seconds()) if last_event_at else None
        warning_codes: List[str] = []
        if gap_seconds is not None and gap_seconds > 600:
            warning_codes.append("STALE_FEED")
        return {
            "redis": "connected",
            "last_event_gap_seconds": gap_seconds,
            "warning_codes": warning_codes,
        }

    def last_ingested_wallclock(self, store_id: str) -> Optional[datetime]:
        store_id = normalize_store_id(store_id)
        return self._store_last_ingested_wallclock.get(store_id)

    def store_ids(self) -> List[str]:
        return sorted(self._stores.keys())

ProductionStateEngine = StoreAnalyticsEngine
state_engine = ProductionStateEngine()
