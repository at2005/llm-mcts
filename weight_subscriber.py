import json
from typing import Callable

from redis import Redis


def run_weight_update_subscriber(
    redis_host: str,
    redis_port: int,
    redis_db: int,
    sync_weights: Callable[[int], None],
    get_current_version: Callable[[], int],
    set_current_version: Callable[[int], None],
    on_version_applied: Callable[[int, int], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
):
    redis = Redis(host=redis_host, port=redis_port, db=redis_db)
    try:
        latest_raw = redis.get("weights:latest_version")
        if latest_raw is not None:
            latest_version = int(latest_raw)
            if latest_version > get_current_version():
                sync_weights(latest_version)
                previous_version = get_current_version()
                set_current_version(latest_version)
                if on_version_applied is not None:
                    on_version_applied(previous_version, latest_version)
    except Exception as e:
        if on_error is not None:
            on_error(e)

    pubsub = redis.pubsub()
    pubsub.subscribe("weights:updates")

    for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            payload = json.loads(message["data"])
            version = int(payload.get("version", 0))

            while True:
                pending = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.0)
                if pending is None:
                    break
                if pending.get("type") != "message":
                    continue
                pending_payload = json.loads(pending["data"])
                version = max(version, int(pending_payload.get("version", 0)))

            latest_raw = redis.get("weights:latest_version")
            if latest_raw is not None:
                version = max(version, int(latest_raw))

            current_version = get_current_version()
            if version <= current_version:
                continue

            previous_version = current_version
            sync_weights(version)
            set_current_version(version)

            if on_version_applied is not None:
                on_version_applied(previous_version, version)
        except Exception as e:
            if on_error is not None:
                on_error(e)
                continue
            raise
