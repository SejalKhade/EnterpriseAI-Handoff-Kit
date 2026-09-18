"""
EnterpriseAI Handoff Kit — S3 Writer
Uploads health reports to S3 when a bucket is configured.
Falls back to the local artifact directory otherwise.

Key layout:
  s3://{bucket}/{prefix}{YYYY}/{MM}/{DD}/health_report_{timestamp}.json
"""

from __future__ import annotations
from datetime import datetime, timezone
import json

from src.config import settings


class S3Writer:
    """Writes JSON artifacts to S3, or to disk when S3 is not configured."""

    def __init__(self):
        self.enabled = settings.s3_configured
        self.bucket = settings.S3_BUCKET
        self.prefix = settings.S3_PREFIX
        self.local_dir = settings.ARTIFACT_DIR
        self._client = None
        self._error: str | None = None

    def _get_client(self):
        if self._client is None:
            import boto3  # lazy import
            self._client = boto3.client("s3", region_name=settings.AWS_REGION)
        return self._client

    @staticmethod
    def _build_key(prefix: str, name: str) -> str:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}{now:%Y/%m/%d}/{name}_{stamp}.json"

    def write_report(self, report: dict, name: str = "health_report") -> dict:
        body = json.dumps(report, indent=2, default=str)
        key = self._build_key(self.prefix, name)

        if not self.enabled:
            return self._write_local(body, key, "S3_BUCKET not configured")

        try:
            self._get_client().put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            return {
                "status": "ok",
                "target": f"s3://{self.bucket}/{key}",
                "bytes":  len(body),
            }
        except Exception as exc:
            self._error = str(exc)
            return self._write_local(body, key, str(exc))

    def _write_local(self, body: str, key: str, reason: str) -> dict:
        path = self.local_dir / key.replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return {
            "status": "fallback",
            "target": str(path),
            "bytes":  len(body),
            "reason": reason,
        }

    def list_reports(self, limit: int = 20) -> list[dict]:
        if not self.enabled:
            files = sorted(self.local_dir.glob("*health_report*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            return [{
                "key":  f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
                "source": "local",
            } for f in files]
        try:
            resp = self._get_client().list_objects_v2(
                Bucket=self.bucket, Prefix=self.prefix, MaxKeys=limit
            )
            return [{
                "key":      o["Key"],
                "size":     o["Size"],
                "modified": o["LastModified"].isoformat(),
                "source":   "s3",
            } for o in resp.get("Contents", [])]
        except Exception as exc:
            self._error = str(exc)
            return []

    def target_info(self) -> dict:
        return {
            "enabled":   self.enabled,
            "bucket":    self.bucket,
            "prefix":    self.prefix,
            "region":    settings.AWS_REGION,
            "fallback":  str(self.local_dir),
        }
