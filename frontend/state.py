from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppState:
    project_root: Path
    image_path: str = ""
    output_path: str = ""
    dictionary_train_image_path: str = ""
    last_result: dict[str, Any] | None = None
    last_result_prev: dict[str, Any] | None = None  # avant-dernière reconstruction (comparaison dico)
    last_sweep: dict[str, Any] | None = None
    last_section6_dir: str = ""
    last_analyses_empreinte: dict[str, Any] | None = None
    last_patch_dir: str = ""
    logs: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-300:]