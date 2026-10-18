"""Runtime configuration and the path half of the privacy boundary.

Every path the system reads or writes is resolved here, so the boundary has exactly one place to be
enforced: real data is read from ``STRATA_DATA_DIR``, private outputs go to ``STRATA_PRIVATE_DIR``,
and neither may resolve inside the repository. Nothing in the code base names the private location;
it is only ever known through the environment.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PRIVATE_SUBDIR = "private_out"
"""Private output folder created next to the extract when ``STRATA_PRIVATE_DIR`` is unset."""

SYNTHETIC_MARKER = "SYNTHETIC.json"
"""File written by the public generator. A data directory without it is treated as real."""


class PrivacyBoundaryError(RuntimeError):
    """Raised when an operation would move real data, or anything derived from it, into the repo."""


class Zone(enum.StrEnum):
    """Which side of the boundary a run is on.

    ``public`` runs read synthetic data and write under ``.strata/`` (gitignored). ``private`` runs
    read the real extract and write only under ``STRATA_PRIVATE_DIR``. The zone is chosen
    explicitly and never inferred from the presence of data, so a forgotten flag fails closed:
    a public run refuses any data directory that does not carry the synthetic marker.
    """

    PUBLIC = "public"
    PRIVATE = "private"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the project that contains this package (or ``start``).

    The package is installed in editable mode, so walking up from its own file finds the project's
    ``pyproject.toml``. Inside a container or a wheel install there is no repository, and the
    current directory is the safest stand-in.
    """
    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and 'name = "strata"' in pyproject.read_text(encoding="utf-8"):
            return candidate
    return Path.cwd().resolve()


def is_within(path: Path, root: Path) -> bool:
    """True when ``path`` resolves to ``root`` or below it (symlinks and ``..`` resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Workspace:
    """Where one zone keeps its artefacts. Built by :meth:`Settings.workspace`."""

    zone: Zone
    root: Path
    warehouse: Path
    ledger_dir: Path
    reports_dir: Path
    figures_dir: Path
    dbt_target_dir: Path
    dbt_log_dir: Path

    @property
    def is_private(self) -> bool:
        return self.zone is Zone.PRIVATE


def default_env_files(root: Path | None = None) -> tuple[str, ...]:
    """Local configuration files read by :class:`Settings`, lowest precedence first.

    ``.env.local`` holds this machine's private locations (``STRATA_DATA_DIR``,
    ``STRATA_PRIVATE_DIR``) so that private runs need no manual export. Both files are gitignored
    and anchored at the repository root, so a command run from a subfolder reads the same file.
    Real environment variables override them.
    """
    base = root or find_repo_root()
    return (str(base / ".env"), str(base / ".env.local"))


ENV_FILES = default_env_files()


class Settings(BaseSettings):
    """Typed view of the ``STRATA_*`` environment.

    Relative paths are resolved against the repository root, never against the current directory,
    so a command run from a subfolder cannot scatter files.
    """

    model_config = SettingsConfigDict(
        env_prefix="STRATA_",
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path | None = Field(default=None, description="Real extract (STRATA_DATA_DIR).")
    private_dir: Path | None = Field(
        default=None,
        description="Private outputs (STRATA_PRIVATE_DIR); default: next to the extract.",
    )
    db: Path = Field(default=Path(".strata/strata.duckdb"), description="Public warehouse file.")
    work_dir: Path = Field(default=Path(".strata"), description="Public scratch space, gitignored.")
    zone: Zone = Field(default=Zone.PUBLIC, description="Default zone (STRATA_ZONE).")
    seed: int = Field(default=42, description="Master seed for every random draw.")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)
    repo_root: Path = Field(default_factory=find_repo_root)

    @model_validator(mode="after")
    def _resolve_and_guard(self) -> Self:
        root = self.repo_root.resolve()
        self.repo_root = root
        if not self.db.is_absolute():
            self.db = root / self.db
        if not self.work_dir.is_absolute():
            self.work_dir = root / self.work_dir
        if self.data_dir is not None:
            self.data_dir = self.data_dir.expanduser().resolve()
            if is_within(self.data_dir, root):
                msg = "STRATA_DATA_DIR resolves inside the repository; real data must stay outside."
                raise PrivacyBoundaryError(msg)
        if self.private_dir is not None:
            self.private_dir = self.private_dir.expanduser().resolve()
            if is_within(self.private_dir, root):
                msg = "STRATA_PRIVATE_DIR resolves inside the repository; private outputs must not."
                raise PrivacyBoundaryError(msg)
        return self

    @property
    def resolved_private_dir(self) -> Path:
        """The private output folder, or an error explaining how to configure one."""
        if self.private_dir is not None:
            return self.private_dir
        if self.data_dir is not None:
            return self.data_dir / PRIVATE_SUBDIR
        msg = "Private runs need STRATA_PRIVATE_DIR or STRATA_DATA_DIR to be set."
        raise PrivacyBoundaryError(msg)

    def require_data_dir(self) -> Path:
        """The real extract folder, or an error naming the variable to set."""
        if self.data_dir is None:
            msg = "STRATA_DATA_DIR is not set; the real extract is reachable only through it."
            raise PrivacyBoundaryError(msg)
        return self.data_dir

    def workspace(self, zone: Zone | None = None) -> Workspace:
        """Artefact locations for ``zone`` (default: the configured zone)."""
        zone = zone or self.zone
        if zone is Zone.PRIVATE:
            root = self.resolved_private_dir
            return Workspace(
                zone=zone,
                root=root,
                warehouse=root / "warehouse" / "strata.duckdb",
                ledger_dir=root / "ledger",
                reports_dir=root / "reports",
                figures_dir=root / "figures",
                dbt_target_dir=root / "dbt" / "target",
                dbt_log_dir=root / "dbt" / "logs",
            )
        work = self.work_dir
        return Workspace(
            zone=zone,
            root=work,
            warehouse=self.db,
            ledger_dir=work / "ledger",
            reports_dir=work / "reports",
            figures_dir=work / "figures",
            dbt_target_dir=work / "dbt" / "target",
            dbt_log_dir=work / "dbt" / "logs",
        )

    def assert_private_output(self, path: Path) -> Path:
        """Return ``path`` resolved if it may receive something derived from real rows.

        Two conditions, both required: outside the repository, and under the private folder. The
        second stops a private artefact from landing in some arbitrary place on the machine.
        """
        resolved = path.expanduser().resolve()
        if is_within(resolved, self.repo_root):
            msg = "Refusing to write a private artefact inside the repository."
            raise PrivacyBoundaryError(msg)
        if not is_within(resolved, self.resolved_private_dir):
            msg = "Private artefacts must be written under STRATA_PRIVATE_DIR."
            raise PrivacyBoundaryError(msg)
        return resolved


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests build their own ``Settings`` instead of using this cache."""
    return Settings()
