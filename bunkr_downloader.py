#!/usr/bin/env python3
import argparse
import base64
import html
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse

import requests


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
RETRY_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
CHUNK_SIZE = 1024 * 512


class RetryableHTTPError(Exception):
    pass


class DeadLinkError(Exception):
    pass


@dataclass
class FileJob:
    slug: str
    page_url: str
    display_name: Optional[str] = None
    index: int = 0


def enable_ansi_colors() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        if kernel32.SetConsoleMode(handle, mode.value | 0x0004) == 0:
            return False
        return True
    except Exception:
        return False


class TerminalUI:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"

    def __init__(self, pretty: bool, workers: int):
        self.pretty = pretty
        self.workers = max(1, workers)
        self.is_tty = sys.stdout.isatty()
        self.dashboard = pretty and self.is_tty and self.workers > 1
        self.dynamic = pretty and self.is_tty and self.workers == 1
        self.compact_multi = (self.workers > 1) and (not self.dashboard)
        self.use_color = pretty and enable_ansi_colors()
        self.lock = threading.Lock()
        self.last_progress_at: dict[str, float] = {}
        self.progress_state: dict[str, dict[str, float]] = {}
        self.term_width = shutil.get_terminal_size((120, 20)).columns
        self.dynamic_active = False
        self.dashboard_slots: list[str] = []
        self.dashboard_key_to_slot: dict[str, int] = {}
        self.dashboard_rendered = False
        self.multi_pct_step = 25  # print every 25% in multi-worker mode
        self.multi_interval = 15.0  # or at least once every N seconds
        self.multi_unknown_step_bytes = 300 * 1024 * 1024  # 300MB

    def _truncate(self, text: str) -> str:
        if len(text) <= self.term_width - 1:
            return text
        return text[: self.term_width - 1]

    def _dashboard_ensure(self) -> None:
        if self.dashboard_slots:
            return
        self.dashboard_slots = [""] * self.workers
        for _ in range(self.workers):
            sys.stdout.write("\n")
        sys.stdout.flush()
        self.dashboard_rendered = True

    def _dashboard_render(self) -> None:
        if not self.dashboard:
            return
        self._dashboard_ensure()
        sys.stdout.write(f"\033[{self.workers}A")
        for i in range(self.workers):
            line = self._truncate(self.dashboard_slots[i] if self.dashboard_slots[i] else "")
            sys.stdout.write("\r" + line.ljust(self.term_width) + "\n")
        sys.stdout.flush()

    def _dashboard_assign_slot(self, key: str, title: str) -> None:
        if key in self.dashboard_key_to_slot:
            idx = self.dashboard_key_to_slot[key]
            self.dashboard_slots[idx] = self._truncate(title)
            return
        used = set(self.dashboard_key_to_slot.values())
        idx = None
        for i in range(self.workers):
            if i not in used:
                idx = i
                break
        if idx is None:
            # Safety fallback; should rarely happen when pool size == workers.
            idx = 0
        self.dashboard_key_to_slot[key] = idx
        self.dashboard_slots[idx] = self._truncate(title)

    def start_task(self, key: str, title: str) -> None:
        with self.lock:
            if not self.dashboard:
                return
            self._dashboard_assign_slot(key, title)
            self._dashboard_render()

    def complete_task(self, key: str, ok: bool, message: str) -> None:
        with self.lock:
            if self.dashboard:
                idx = self.dashboard_key_to_slot.pop(key, None)
                if idx is not None:
                    status = "DONE" if ok else "FAIL"
                    self.dashboard_slots[idx] = self._truncate(f"[{status}] {message}")
                    self._dashboard_render()
                    self.dashboard_slots[idx] = ""
                    self._dashboard_render()
                return
        if ok:
            self.ok(message)
        else:
            self.error(message)

    def _color(self, text: str, color: str) -> str:
        if not self.use_color:
            return text
        return f"{color}{text}{self.RESET}"

    def _line(self, text: str) -> None:
        with self.lock:
            if self.dashboard:
                self._dashboard_ensure()
                sys.stdout.write(f"\033[{self.workers}A")
                print(text, flush=True)
                for _ in range(self.workers - 1):
                    sys.stdout.write("\n")
                self._dashboard_render()
                return
            if self.dynamic and self.dynamic_active:
                sys.stdout.write("\n")
                self.dynamic_active = False
            print(text, flush=True)

    def info(self, msg: str) -> None:
        self._line(self._color("[INFO]", self.CYAN) + f" {msg}")

    def ok(self, msg: str) -> None:
        self._line(self._color("[ OK ]", self.GREEN) + f" {msg}")

    def warn(self, msg: str) -> None:
        self._line(self._color("[WARN]", self.YELLOW) + f" {msg}")

    def error(self, msg: str) -> None:
        self._line(self._color("[FAIL]", self.RED) + f" {msg}")

    def _render_bar(self, current: int, total: Optional[int], width: int = 22) -> str:
        if not total or total <= 0:
            return "[" + ("." * width) + "]"
        pct = max(0.0, min(1.0, current / total))
        fill = int(width * pct)
        return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"

    def progress(
        self,
        key: str,
        label: str,
        current: int,
        total: Optional[int],
        speed_bps: float,
        attempt: int,
        resumed: bool,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        last = self.last_progress_at.get(key, 0.0)
        min_interval = 0.20 if self.dynamic else 0.45
        if not force and (now - last) < min_interval:
            return

        if self.compact_multi and not force:
            state = self.progress_state.setdefault(
                key,
                {"pct_bucket": -1.0, "byte_bucket": -1.0, "last_ts": 0.0},
            )
            should_print = False
            if total and total > 0:
                pct_now = (current / total) * 100.0
                bucket = float(int(pct_now // self.multi_pct_step))
                if bucket > state["pct_bucket"]:
                    state["pct_bucket"] = bucket
                    should_print = True
            else:
                bucket = float(int(current // self.multi_unknown_step_bytes))
                if bucket > state["byte_bucket"]:
                    state["byte_bucket"] = bucket
                    should_print = True

            if not should_print and (now - state["last_ts"]) < self.multi_interval:
                return
            state["last_ts"] = now

        self.last_progress_at[key] = now

        pct = (current / total * 100.0) if total and total > 0 else 0.0
        bar_width = 12 if self.compact_multi else 22
        bar = self._render_bar(current, total, width=bar_width)
        total_str = human_bytes(total) if total else "?"
        retry_part = f" retry:{attempt}" if attempt > 0 else ""
        resume_part = " resume" if resumed else ""
        line = (
            f"{label:<30} {bar} {pct:6.2f}% "
            f"{human_bytes(current):>10}/{total_str:<10} "
            f"{human_bytes(speed_bps):>8}/s{resume_part}{retry_part}"
        )
        line = self._truncate(line)

        with self.lock:
            if self.dashboard:
                self._dashboard_assign_slot(key, line)
                self._dashboard_render()
                return
            if self.dynamic:
                sys.stdout.write("\r" + line.ljust(self.term_width))
                sys.stdout.flush()
                self.dynamic_active = True
            else:
                print(line, flush=True)

    def finish_progress_line(self) -> None:
        with self.lock:
            if self.dashboard:
                return
            if self.dynamic and self.dynamic_active:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.dynamic_active = False


class SessionFactory:
    def __init__(self):
        self.local = threading.local()

    def get(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            self.local.session = session
        return session


class DownloadArchive:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
                slug = line.strip()
                if slug:
                    self.seen.add(slug)

    def has(self, slug: str) -> bool:
        return slug in self.seen

    def add(self, slug: str) -> None:
        with self.lock:
            if slug in self.seen:
                return
            with self.path.open("a", encoding="utf-8") as f:
                f.write(slug + "\n")
            self.seen.add(slug)


class FailedLinkLogger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.header_written = path.exists() and path.stat().st_size > 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

    def add(
        self,
        target_url: str,
        slug: str,
        filename: str,
        reason: str,
        resolved_url: Optional[str],
        file_page_url: Optional[str],
    ) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fields = [
            ts,
            self._safe(target_url),
            self._safe(slug),
            self._safe(filename),
            self._safe(reason),
            self._safe(resolved_url),
            self._safe(file_page_url),
        ]
        line = "\t".join(fields) + "\n"
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                if not self.header_written:
                    f.write(
                        "timestamp\ttarget_url\tslug\tfilename\treason\tresolved_url\tfile_page_url\n"
                    )
                    self.header_written = True
                f.write(line)


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


def parse_target_inputs(input_value: str) -> list[str]:
    candidate = Path(input_value)
    if candidate.exists() and candidate.is_file():
        return parse_urls_file(candidate)
    return [ensure_url(input_value)]


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


def canonicalize_url(url: str) -> str:
    p = urlparse(url)
    query = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(p.query, keep_blank_values=True)))
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, "", query, ""))


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    retries: int,
    **kwargs,
) -> requests.Response:
    for attempt in range(retries + 1):
        try:
            resp = session.request(method=method, url=url, timeout=timeout, **kwargs)
            if resp.status_code in RETRY_HTTP_STATUS:
                status = resp.status_code
                resp.close()
                raise RetryableHTTPError(f"Retryable HTTP status: {status}")
            return resp
        except (requests.RequestException, RetryableHTTPError):
            if attempt >= retries:
                raise
            wait = min(20.0, 1.25 * (2**attempt) + random.uniform(0.1, 0.45))
            time.sleep(wait)
    raise RuntimeError("unreachable")


def extract_slugs_from_links(album_html: str) -> list[str]:
    pattern = re.compile(
        r'href=["\'](?:https?://[^"\']+)?/f/([A-Za-z0-9]+)["\']',
        re.IGNORECASE,
    )
    return [m.group(1) for m in pattern.finditer(album_html)]


def extract_album_file_map(album_html: str) -> dict[str, str]:
    file_map: dict[str, str] = {}
    m = re.search(r"window\.albumFiles\s*=\s*(\[[\s\S]*?\]);", album_html)
    if not m:
        return file_map
    raw = m.group(1)
    try:
        payload = json.loads(raw)
    except Exception:
        return file_map
    if not isinstance(payload, list):
        return file_map
    for item in payload:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        name = item.get("original") or item.get("name")
        if isinstance(slug, str) and slug and isinstance(name, str) and name:
            file_map[slug] = name
    return file_map


def extract_album_page_links(
    album_html: str,
    current_page_url: str,
    album_path: str,
) -> list[str]:
    links = set()
    parsed_current = urlparse(current_page_url)
    current_origin = f"{parsed_current.scheme}://{parsed_current.netloc}"
    base_album_path = album_path.rstrip("/")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', album_html, flags=re.IGNORECASE)
    for href in hrefs:
        full = urljoin(current_page_url, html.unescape(href))
        p = urlparse(full)
        if p.scheme not in {"http", "https"}:
            continue
        if f"{p.scheme}://{p.netloc}" != current_origin:
            continue
        path = p.path.rstrip("/")
        if "/f/" in path:
            continue
        if not (path == base_album_path or path.startswith(base_album_path + "/")):
            continue
        if path == base_album_path and not p.query:
            continue
        links.add(urlunparse((p.scheme, p.netloc, p.path, "", p.query, "")))
    return sorted(links)


def extract_filename_from_file_page(file_html: str) -> Optional[str]:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", file_html, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    text = re.sub(r"<[^>]+>", "", m.group(1))
    text = html.unescape(text).strip()
    return text or None


def extract_download_link_from_file_page(file_html: str, file_page_url: str) -> Optional[str]:
    m = re.search(
        r'href=["\']([^"\']+)["\'][^>]*>\s*Download\s*<',
        file_html,
        re.IGNORECASE,
    )
    if not m:
        return None
    return urljoin(file_page_url, m.group(1))


def decrypt_vs_url(encoded_url: str, timestamp: int) -> str:
    key = f"SECRET_KEY_{timestamp // 3600}".encode("utf-8")
    raw = base64.b64decode(encoded_url)
    out = bytes(raw[i] ^ key[i % len(key)] for i in range(len(raw)))
    return out.decode("utf-8")


def resolve_direct_url(
    session: requests.Session,
    origin: str,
    slug: str,
    timeout: int,
    retries: int,
) -> Optional[str]:
    api_url = urljoin(origin, "/api/vs")
    try:
        r = request_with_retry(
            session=session,
            method="POST",
            url=api_url,
            timeout=timeout,
            retries=retries,
            json={"slug": slug},
        )
        r.raise_for_status()
        data = r.json()
        vs_url = data.get("url")
        if not vs_url:
            return None
        if data.get("encrypted"):
            timestamp = int(data["timestamp"])
            return decrypt_vs_url(vs_url, timestamp)
        return urljoin(origin, str(vs_url))
    except Exception:
        return None


def resolve_metadata(
    session: requests.Session,
    origin: str,
    job: FileJob,
    timeout: int,
    retries: int,
) -> tuple[Optional[str], Optional[str]]:
    direct_url = resolve_direct_url(session, origin, job.slug, timeout=timeout, retries=retries)
    filename = None

    try:
        r = request_with_retry(
            session=session,
            method="GET",
            url=job.page_url,
            timeout=timeout,
            retries=retries,
        )
        r.raise_for_status()
        file_html = r.text
        filename = extract_filename_from_file_page(file_html)
        if not direct_url:
            direct_url = extract_download_link_from_file_page(file_html, job.page_url)
    except Exception:
        pass

    if not filename:
        if direct_url:
            filename = os.path.basename(urlparse(direct_url).path) or job.slug
        else:
            filename = job.slug

    if job.display_name and job.display_name != job.slug:
        filename = job.display_name

    return direct_url, filename


def parse_total_from_content_range(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"/(\d+)$", value.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


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

    # Last part is filename, previous parts are folders (if present).
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


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    skip_existing: bool,
    timeout: int,
    retries: int,
    ui: TerminalUI,
    label: str,
    referer: Optional[str] = None,
    origin: Optional[str] = None,
    skip_dead: bool = False,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and skip_existing:
        ui.complete_task(label, True, f"{label} already exists, skipped")
        return "skipped"

    if out_path.exists() and not skip_existing:
        out_path = unique_path(out_path)

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    attempt = 0
    resumed_once = tmp_path.exists() and tmp_path.stat().st_size > 0
    speed_bps = 0.0

    while True:
        current_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers: dict[str, str] = {}
        if referer:
            headers["Referer"] = referer
        if origin:
            headers["Origin"] = origin
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"

        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, timeout)) as r:
                if r.status_code in RETRY_HTTP_STATUS:
                    raise RetryableHTTPError(f"Retryable HTTP status: {r.status_code}")
                if skip_dead and r.status_code == 404:
                    raise DeadLinkError(f"404 Not Found: {r.url}")

                if current_size > 0 and r.status_code == 200:
                    # Server ignored Range; restart full download.
                    current_size = 0
                    if tmp_path.exists():
                        tmp_path.unlink()

                if r.status_code == 416:
                    total_416 = parse_total_from_content_range(r.headers.get("Content-Range"))
                    if total_416 is not None and tmp_path.exists() and tmp_path.stat().st_size >= total_416:
                        tmp_path.replace(out_path)
                        ui.complete_task(label, True, f"{label} completed from previous partial file")
                        return "downloaded"
                    current_size = 0
                    if tmp_path.exists():
                        tmp_path.unlink()
                    raise RetryableHTTPError("Range not satisfiable, restarting")

                r.raise_for_status()

                total = None
                if r.status_code == 206:
                    total = parse_total_from_content_range(r.headers.get("Content-Range"))
                    if total is None:
                        content_len = int(r.headers.get("Content-Length", "0") or "0")
                        total = current_size + content_len if content_len > 0 else None
                else:
                    content_len = int(r.headers.get("Content-Length", "0") or "0")
                    total = content_len if content_len > 0 else None

                mode = "ab" if current_size > 0 else "wb"
                downloaded = current_size
                last_tick_at = time.monotonic()
                last_tick_bytes = downloaded
                with tmp_path.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        delta_t = now - last_tick_at
                        if delta_t >= 0.25:
                            speed_bps = (downloaded - last_tick_bytes) / delta_t
                            last_tick_at = now
                            last_tick_bytes = downloaded
                            ui.progress(
                                key=label,
                                label=label,
                                current=downloaded,
                                total=total,
                                speed_bps=speed_bps,
                                attempt=attempt,
                                resumed=resumed_once,
                            )

                if total is not None and downloaded < total:
                    raise RetryableHTTPError(
                        f"Incomplete stream ({downloaded}/{total}), retrying..."
                    )

                ui.progress(
                    key=label,
                    label=label,
                    current=downloaded,
                    total=total or downloaded,
                    speed_bps=speed_bps,
                    attempt=attempt,
                    resumed=resumed_once,
                    force=True,
                )
                tmp_path.replace(out_path)
                ui.finish_progress_line()
                ui.complete_task(label, True, f"{label} saved -> {out_path.name}")
                return "downloaded"
        except DeadLinkError:
            ui.finish_progress_line()
            raise
        except (requests.RequestException, RetryableHTTPError) as exc:
            if attempt >= retries:
                ui.finish_progress_line()
                raise RuntimeError(str(exc)) from exc
            attempt += 1
            wait = min(20.0, 1.5 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5))
            resumed_once = resumed_once or (tmp_path.exists() and tmp_path.stat().st_size > 0)
            if ui.dashboard:
                ui.progress(
                    key=label,
                    label=f"{label} retry {attempt}/{retries} in {wait:.1f}s",
                    current=current_size,
                    total=None,
                    speed_bps=0.0,
                    attempt=attempt,
                    resumed=resumed_once,
                    force=True,
                )
            else:
                ui.warn(f"{label} network issue, retry {attempt}/{retries} in {wait:.1f}s")
            time.sleep(wait)


def collect_album_jobs(
    album_url: str,
    session: requests.Session,
    timeout: int,
    retries: int,
    max_pages: int,
) -> list[FileJob]:
    parsed = urlparse(album_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    album_path_match = re.match(r"^(/a/[A-Za-z0-9]+)", parsed.path)
    album_path = album_path_match.group(1) if album_path_match else parsed.path.rstrip("/")

    queue = deque([album_url])
    visited: set[str] = set()
    seen_slugs: set[str] = set()
    display_name_map: dict[str, str] = {}

    pages_checked = 0
    while queue and pages_checked < max_pages:
        page_url = queue.popleft()
        page_key = canonicalize_url(page_url)
        if page_key in visited:
            continue
        visited.add(page_key)
        pages_checked += 1

        r = request_with_retry(
            session=session,
            method="GET",
            url=page_url,
            timeout=timeout,
            retries=retries,
        )
        r.raise_for_status()
        html_text = r.text

        payload_map = extract_album_file_map(html_text)
        display_name_map.update(payload_map)

        for slug in extract_slugs_from_links(html_text):
            seen_slugs.add(slug)

        for slug in payload_map:
            seen_slugs.add(slug)

        for link in extract_album_page_links(html_text, page_url, album_path):
            link_key = canonicalize_url(link)
            if link_key not in visited:
                queue.append(link)

    jobs = [
        FileJob(
            slug=slug,
            page_url=urljoin(origin, f"/f/{slug}"),
            display_name=display_name_map.get(slug),
        )
        for slug in seen_slugs
    ]
    return jobs


def build_jobs(
    url: str,
    session: requests.Session,
    timeout: int,
    retries: int,
    max_pages: int,
) -> tuple[str, list[FileJob]]:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path

    m_file = re.match(r"^/f/([A-Za-z0-9]+)", path)
    if m_file:
        slug = m_file.group(1)
        return origin, [FileJob(slug=slug, page_url=urljoin(origin, f"/f/{slug}"))]

    m_album = re.match(r"^/a/([A-Za-z0-9]+)", path)
    if not m_album:
        raise RuntimeError("URL must be a Bunkr album (/a/...) or file (/f/...) link.")

    jobs = collect_album_jobs(
        album_url=url,
        session=session,
        timeout=timeout,
        retries=retries,
        max_pages=max_pages,
    )
    if not jobs:
        raise RuntimeError("No file links found on album page.")
    return origin, jobs


def worker(
    sessions: SessionFactory,
    ui: TerminalUI,
    origin: str,
    total_jobs: int,
    job: FileJob,
    out_dir: Path,
    target_prefix: str,
    target_url: str,
    dry_run: bool,
    skip_existing: bool,
    timeout: int,
    retries: int,
    archive: Optional[DownloadArchive],
    failed_logger: Optional[FailedLinkLogger],
    skip_dead: bool,
) -> str:
    session = sessions.get()
    prefix = f"{target_prefix} " if target_prefix else ""
    label = f"{prefix}[{job.index:>3}/{total_jobs}] {job.slug}"
    ui.start_task(label, f"{label} resolving...")

    direct_url, raw_name = resolve_metadata(
        session=session,
        origin=origin,
        job=job,
        timeout=timeout,
        retries=retries,
    )
    if not direct_url:
        if failed_logger:
            failed_logger.add(
                target_url=target_url,
                slug=job.slug,
                filename=raw_name or job.slug,
                reason="cannot resolve file URL",
                resolved_url=None,
                file_page_url=job.page_url,
            )
        ui.complete_task(label, False, f"{label} cannot resolve file URL")
        return "failed"

    out_path = build_target_path(
        base_dir=out_dir,
        raw_name=raw_name or job.slug,
        fallback_slug=job.slug,
        direct_url=direct_url,
    )
    safe_name = str(out_path.relative_to(out_dir)).replace("\\", "/")

    if not ui.dashboard:
        ui.info(f"{label} -> {safe_name}")
    if dry_run:
        ui.complete_task(label, True, f"{label} resolved")
        if not ui.dashboard:
            ui.info(f"{label} direct URL: {direct_url}")
        return "resolved"

    try:
        status = download_file(
            session=session,
            url=direct_url,
            out_path=out_path,
            skip_existing=skip_existing,
            timeout=timeout,
            retries=retries,
            ui=ui,
            label=label,
            referer=job.page_url,
            origin=origin,
            skip_dead=skip_dead,
        )
        if archive and status in {"downloaded", "skipped"}:
            archive.add(job.slug)
        return status
    except DeadLinkError as exc:
        if failed_logger:
            failed_logger.add(
                target_url=target_url,
                slug=job.slug,
                filename=safe_name,
                reason=str(exc),
                resolved_url=direct_url,
                file_page_url=job.page_url,
            )
        ui.complete_task(label, False, f"{label} {exc}")
        return "failed"
    except Exception as exc:
        if failed_logger:
            failed_logger.add(
                target_url=target_url,
                slug=job.slug,
                filename=safe_name,
                reason=str(exc),
                resolved_url=direct_url,
                file_page_url=job.page_url,
            )
        ui.complete_task(label, False, f"{label} {exc}")
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from Bunkr album/file URLs.",
    )
    parser.add_argument(
        "input",
        help="Single URL or text file path (1 URL per line, e.g. urls.text)",
    )
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Parallel downloads")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--max-pages", type=int, default=50, help="Max album pages to crawl")
    parser.add_argument("--dry-run", action="store_true", help="Resolve links only, do not download")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for network errors")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing filenames; save as 'name (1).ext'",
    )
    parser.add_argument("--no-archive", action="store_true", help="Disable downloaded archive file")
    parser.add_argument(
        "--archive-file",
        default="",
        help="Path to archive file (default: <output>/downloaded.txt)",
    )
    parser.add_argument(
        "--failed-file",
        default="failed_links.txt",
        help="Filename/path for failed links log (default: failed_links.txt in output root)",
    )
    parser.add_argument(
        "--skip-dead",
        action="store_true",
        help="If final download response is 404, skip immediately without retries",
    )
    parser.add_argument("--no-pretty", action="store_true", help="Disable pretty terminal output")
    return parser.parse_args()


def run_target(
    target_url: str,
    args: argparse.Namespace,
    ui: TerminalUI,
    sessions: SessionFactory,
    workers: int,
    retries: int,
    timeout: int,
    skip_existing: bool,
    target_idx: int,
    target_total: int,
    failed_logger: Optional[FailedLinkLogger],
) -> dict[str, int]:
    counts = {"downloaded": 0, "skipped": 0, "failed": 0, "resolved": 0}
    target_prefix = f"[LINK {target_idx}/{target_total}]" if target_total > 1 else ""
    album_id, _ = extract_url_ids(target_url)
    out_dir = Path(args.output)
    if album_id:
        out_dir = out_dir / album_id
    out_dir.mkdir(parents=True, exist_ok=True)

    session = sessions.get()
    try:
        origin, jobs = build_jobs(
            url=target_url,
            session=session,
            timeout=timeout,
            retries=retries,
            max_pages=max(1, args.max_pages),
        )
    except Exception as exc:
        ui.error(f"Error while reading page: {exc}")
        counts["failed"] += 1
        return counts

    if args.max_files > 0:
        jobs = jobs[: args.max_files]

    jobs = sorted(jobs, key=lambda j: j.slug)
    for i, job in enumerate(jobs, start=1):
        job.index = i

    archive = None
    if not args.no_archive and not args.dry_run:
        archive_path = Path(args.archive_file) if args.archive_file else (out_dir / "downloaded.txt")
        archive = DownloadArchive(archive_path)
        before = len(jobs)
        jobs = [job for job in jobs if not archive.has(job.slug)]
        skipped = before - len(jobs)
        if skipped > 0:
            ui.info(f"Skipped {skipped} file(s) already in archive")
        for i, job in enumerate(jobs, start=1):
            job.index = i

    if target_prefix:
        ui.info(f"{target_prefix} Found {len(jobs)} file(s)")
    else:
        ui.info(f"Found {len(jobs)} file(s)")
    if not jobs:
        return counts

    total_jobs = len(jobs)
    if workers == 1:
        for job in jobs:
            status = worker(
                sessions=sessions,
                ui=ui,
                origin=origin,
                total_jobs=total_jobs,
                job=job,
                out_dir=out_dir,
                target_prefix=target_prefix,
                target_url=target_url,
                dry_run=args.dry_run,
                skip_existing=skip_existing,
                timeout=timeout,
                retries=retries,
                archive=archive,
                failed_logger=failed_logger,
                skip_dead=args.skip_dead,
            )
            counts[status] = counts.get(status, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    worker,
                    sessions,
                    ui,
                    origin,
                    total_jobs,
                    job,
                    out_dir,
                    target_prefix,
                    target_url,
                    args.dry_run,
                    skip_existing,
                    timeout,
                    retries,
                    archive,
                    failed_logger,
                    args.skip_dead,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                status = "failed"
                try:
                    status = future.result()
                except Exception as exc:
                    ui.error(f"Worker error: {exc}")
                counts[status] = counts.get(status, 0) + 1

    return counts


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    input_is_file = input_path.exists() and input_path.is_file()
    try:
        targets = parse_target_inputs(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    skip_existing = not args.no_skip_existing
    workers = max(1, args.workers)
    retries = max(0, args.retries)
    timeout = max(10, args.timeout)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    ui = TerminalUI(pretty=not args.no_pretty, workers=workers)
    sessions = SessionFactory()
    failed_path = Path(args.failed_file)
    if not failed_path.is_absolute():
        failed_path = output_root / failed_path
    failed_logger = FailedLinkLogger(failed_path)
    overall = {"downloaded": 0, "skipped": 0, "failed": 0, "resolved": 0}
    multi_target = len(targets) > 1
    if input_is_file:
        ui.info(f"Loaded {len(targets)} link(s) from {input_path.name}")

    for idx, target_url in enumerate(targets, start=1):
        if multi_target:
            ui.info(f"[LINK {idx}/{len(targets)}] {target_url}")
        counts = run_target(
            target_url=target_url,
            args=args,
            ui=ui,
            sessions=sessions,
            workers=workers,
            retries=retries,
            timeout=timeout,
            skip_existing=skip_existing,
            target_idx=idx,
            target_total=len(targets),
            failed_logger=failed_logger,
        )
        ui.finish_progress_line()
        summary_prefix = f"[LINK {idx}/{len(targets)}] " if multi_target else ""
        ui.info(
            f"{summary_prefix}Summary: "
            f"downloaded={counts.get('downloaded', 0)}, "
            f"skipped={counts.get('skipped', 0)}, "
            f"failed={counts.get('failed', 0)}, "
            f"resolved={counts.get('resolved', 0)}"
        )
        for k, v in counts.items():
            overall[k] = overall.get(k, 0) + v

    if multi_target:
        ui.info(
            "Overall: "
            f"downloaded={overall.get('downloaded', 0)}, "
            f"skipped={overall.get('skipped', 0)}, "
            f"failed={overall.get('failed', 0)}, "
            f"resolved={overall.get('resolved', 0)}"
        )
    if overall.get("failed", 0) > 0:
        ui.info(f"Failed links saved to: {failed_path}")
    return 0 if overall.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
