import html
import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlparse, urlunparse

from .types import INVALID_FS_CHARS


def clean_filename(name: str, fallback: str) -> str:
    name = html.unescape(name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = INVALID_FS_CHARS.sub("_", name).strip(" .")
    return name or fallback


def clean_path_component(name: str, fallback: str) -> str:
    name = html.unescape(name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = INVALID_FS_CHARS.sub("_", name).strip(" .")
    if not name or name in {".", ".."}:
        return fallback
    return name


def ensure_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("Empty URL")
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    return url


def canonicalize_url(url: str) -> str:
    p = urlparse(url)
    query = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(p.query, keep_blank_values=True)))
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, "", query, ""))


def parse_urls_file(path: Path) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        try:
            normalized = ensure_url(line)
        except ValueError:
            continue
        key = canonicalize_url(normalized)
        if key in seen:
            continue
        seen.add(key)
        urls.append(normalized)
    if not urls:
        raise ValueError(f"No valid URL found in file: {path}")
    return urls


def parse_target_inputs(input_value: str) -> list[str]:
    candidate = Path(input_value)
    if candidate.exists() and candidate.is_file():
        return parse_urls_file(candidate)
    return [ensure_url(input_value)]


def get_origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def extract_url_ids(url: str) -> tuple[Optional[str], Optional[str]]:
    path = urlparse(url).path
    m_album = re.match(r"^/a/([A-Za-z0-9]+)", path)
    if m_album:
        return m_album.group(1), None
    m_file = re.match(r"^/f/([A-Za-z0-9]+)", path)
    if m_file:
        return None, m_file.group(1)
    return None, None


def human_bytes(value: Optional[float]) -> str:
    if value is None:
        return "?"
    value = float(max(0.0, value))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f}{units[idx]}"


def build_target_path(
    base_dir: Path,
    raw_name: str,
    fallback_slug: str,
    direct_url: str,
) -> Path:
    raw_name = html.unescape(raw_name or "").strip()
    if not raw_name:
        raw_name = fallback_slug

    parts = [p for p in re.split(r"[\\/]+", raw_name) if p and p not in {".", ".."}]
    if not parts:
        parts = [fallback_slug]

    dir_parts = [clean_path_component(p, "folder") for p in parts[:-1]]
    file_name = clean_filename(parts[-1], fallback=f"{fallback_slug}.bin")

    if "." not in Path(file_name).name:
        ext = Path(urlparse(direct_url).path).suffix
        if ext:
            file_name += ext

    out = base_dir
    for part in dir_parts:
        out = out / part
    return out / file_name
