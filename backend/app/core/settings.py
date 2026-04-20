from functools import lru_cache
from pathlib import Path
from typing import Any
import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    sendgrid_api_key: str
    cors_origins: str = "http://localhost:5173"
    chroma_dir: str = "./chroma_db"
    config_path: str = "config.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    settings = get_settings()
    backend_root = Path(__file__).resolve().parents[2]
    config_file = backend_root / "config.json"

    if settings.config_path:
        candidate = Path(settings.config_path)
        if candidate.is_file():
            config_file = candidate
        else:
            relative_candidate = backend_root / settings.config_path
            if relative_candidate.is_file():
                config_file = relative_candidate
            else:
                project_relative = backend_root.parent / settings.config_path
                if project_relative.is_file():
                    config_file = project_relative

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)
