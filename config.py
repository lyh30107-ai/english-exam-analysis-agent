"""Read author-managed credentials. This module never logs credentials."""
import os
from pathlib import Path
import tomllib

BASE_DIR = Path(__file__).resolve().parent

def settings():
    config = {}
    for path in [BASE_DIR / "api_keys.toml", BASE_DIR / ".streamlit/secrets.toml"]:
        if path.exists():
            config.update(tomllib.loads(path.read_text(encoding="utf-8-sig")))
    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL") or config.get("OPENAI_MODEL", "gpt-6.1-terra"),
        "effort": os.environ.get("OPENAI_REASONING_EFFORT") or config.get("OPENAI_REASONING_EFFORT", "low"),
        "deadline": 18.0,
    }
