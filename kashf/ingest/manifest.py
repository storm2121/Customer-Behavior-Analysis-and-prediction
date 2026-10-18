"""The manifest: what was loaded, from which bytes, and the fingerprint every claim carries.

The fingerprint is a hash of the manifest's content without its timestamp, so loading the same five
files twice gives the same fingerprint, and changing a single byte of any file changes it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kashf.contracts.extracts import CONTRACT_VERSION


class FileManifest(BaseModel):
    """One loaded file."""

    model_config = ConfigDict(frozen=True)

    extract: str
    filename: str
    sha256: str
    size_bytes: int
    rows: int
    columns: int
    line_terminator: str
    date_min: date | None
    date_max: date | None


class Manifest(BaseModel):
    """Every file of one load, plus where they came from."""

    model_config = ConfigDict(frozen=True)

    contract_version: str = CONTRACT_VERSION
    source_kind: Literal["synthetic", "real"]
    generator: dict[str, str | int] | None = None
    files: list[FileManifest]
    loaded_at: datetime

    @property
    def fingerprint(self) -> str:
        """sha256 of the content that identifies the data (timestamp excluded)."""
        content = self.model_dump(mode="json", exclude={"loaded_at"})
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def file(self, extract: str) -> FileManifest:
        return next(item for item in self.files if item.extract == extract)
