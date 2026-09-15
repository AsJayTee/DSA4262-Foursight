"""Cloudflare R2 access. Imported by scripts/download_data.py only.

Deliberately not imported by m6a.data: predict.py runs on an evaluator's
machine that has neither boto3 nor credentials, and must never need them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MANIFEST_KEY = "manifest.json"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in — "
            "ask in the Telegram groupchat for the shared values."
        )
    return value


def client():
    """An S3 client pointed at R2. Requires the `train` extra (boto3)."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is not installed. Run: pip install -e '.[train]'"
        ) from exc

    return boto3.client(
        "s3",
        endpoint_url=_require("R2_ENDPOINT"),
        aws_access_key_id=_require("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def bucket() -> str:
    return _require("R2_BUCKET")


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(s3=None) -> dict:
    """Fetch manifest.json: {key: {"sha256": ..., "size": ...}}."""
    s3 = s3 or client()
    body = s3.get_object(Bucket=bucket(), Key=MANIFEST_KEY)["Body"].read()
    return json.loads(body)


def download(key: str, dest: Path, expected: dict | None = None, s3=None) -> str:
    """Download one object, skipping it if a matching copy is already present.

    Verifying against the manifest catches truncated downloads immediately
    rather than surfacing an hour later as a baffling JSON parse error.
    Returns one of: "skipped", "downloaded".
    """
    s3 = s3 or client()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected:
        if dest.stat().st_size == expected.get("size") and sha256(dest) == expected.get("sha256"):
            return "skipped"

    s3.download_file(bucket(), key, str(dest))

    if expected:
        got = sha256(dest)
        if got != expected.get("sha256"):
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"{key} downloaded but its checksum does not match the manifest "
                f"(expected {expected.get('sha256')[:12]}..., got {got[:12]}...). "
                "The file was removed; try again."
            )
    return "downloaded"


def list_keys(prefix: str = "", s3=None) -> list[str]:
    s3 = s3 or client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        keys += [obj["Key"] for obj in page.get("Contents", [])]
    return keys
