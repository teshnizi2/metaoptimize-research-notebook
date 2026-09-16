"""Inputs of the Python data tests that live outside this repository.

Most data tests read only the published snapshot under public/ and run anywhere. The rest
also read the maintainer's campaign workspace (outputs/tables/*.csv, work/*.json), the
export audit receipt, the extended run inventory, or the original research repository.
Those inputs are not part of a clone, so a test that needs one is decorated with
``needs(...)``: it runs when the input is configured through its environment variable or
present at its default path, and is skipped with a reason naming the variable otherwise.

A configured variable always runs the test, even if the path is wrong, so a maintainer's
misconfiguration fails loudly instead of being skipped.
"""
from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from pathlib import Path

PORTAL = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExternalInput:
    env: str
    path: Path
    probe: Path
    what: str
    configured: bool

    @property
    def available(self) -> bool:
        return self.configured or self.probe.exists()

    @property
    def reason(self) -> str:
        return f"needs {self.what}: set {self.env} (not configured, and {self.probe} does not exist)"


def external_input(env: str, default: Path, what: str, probe: str | None = None, environ=None, inherited: bool = False) -> ExternalInput:
    """``inherited``: the default lives inside another input the maintainer configured, so it counts as configured too."""
    environ = os.environ if environ is None else environ
    own = bool(environ.get(env))
    path = Path(environ[env]) if own else Path(default)
    return ExternalInput(env, path, path / probe if probe else path, what, own or inherited)


def inputs(environ=None):
    """The four external inputs, resolved from ``environ`` (default: the process environment)."""
    environ = os.environ if environ is None else environ
    workspace = external_input("NOTEBOOK_WORKSPACE", PORTAL.parents[1], "the maintainer's campaign workspace",
                               "outputs/tables/complete_experiment_register.csv", environ)
    repo = external_input("NOTEBOOK_RESEARCH_REPO", Path.home() / "Saber Optimization/alice-backup/hierarchical-metaoptimize",
                          "the original research repository", ".git", environ)
    audit = external_input("NOTEBOOK_DATA_AUDIT", workspace.path / "work/portal_data_audit.json",
                           "the research export's audit receipt", None, environ, workspace.configured)
    inventory = external_input("NOTEBOOK_RUN_INVENTORY", workspace.path / "outputs/tables/complete_run_inventory.csv",
                               "the extended run inventory", None, environ, workspace.configured)
    return workspace, repo, audit, inventory


def needs(*required: ExternalInput):
    """unittest.skipUnless for every missing input, naming the first one that is absent."""
    missing = [item for item in required if not item.available]
    return unittest.skipUnless(not missing, missing[0].reason if missing else "")
