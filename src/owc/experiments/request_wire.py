"""One bounded wire format for runner and external model adapters."""

import json

MAX_REQUEST_BYTES = 256_000
REQUEST_HEADROOM_BYTES = 4_096


class RequestSizeError(ValueError):
    """The complete agent request cannot fit the declared wire budget."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def bounded_request(request: dict) -> tuple[dict, bytes]:
    """Drop oldest compact summaries until the whole request has headroom."""
    wire = canonical_bytes(request)
    while len(wire) > MAX_REQUEST_BYTES - REQUEST_HEADROOM_BYTES and request["previous_rounds"]:
        request["previous_rounds"].pop(0)
        request["previous_rounds_omitted"] += 1
        wire = canonical_bytes(request)
    if len(wire) > MAX_REQUEST_BYTES - REQUEST_HEADROOM_BYTES:
        raise RequestSizeError(
            f"Agent request is {len(wire)} bytes; limit with headroom is "
            f"{MAX_REQUEST_BYTES - REQUEST_HEADROOM_BYTES} bytes"
        )
    return request, wire
