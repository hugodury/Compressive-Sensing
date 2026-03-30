"""
Point d'entrée du backend.
"""

from typing import Any


def run_backend(*args: Any, **kwargs: Any) -> None:
    print("Backend démarré.")


if __name__ == "__main__":
    run_backend()

