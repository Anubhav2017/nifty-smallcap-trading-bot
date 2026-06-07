from pathlib import Path
import os


def repo_root(config_path: Path | None = None) -> Path:
    """Project root for resolving paths in JSON configs under config/."""
    if config_path is not None:
        p = config_path.resolve()
        if p.parent.name == "config":
            return p.parent.parent
        return p.parent
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(config_path: Path, relative: str) -> Path:
    """Resolve a repo-relative path using the config file location."""
    rel = str(relative).strip()
    if not rel:
        raise ValueError("empty path")
    p = Path(rel)
    if p.is_absolute():
        return p
    return (repo_root(config_path) / p).resolve()


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
