import argparse
import base64
import html
import json
import os
import random
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests

from .state import DownloadArchive, FailedLinkLogger, SessionFactory
from .types import CHUNK_SIZE, RETRY_HTTP_STATUS, DeadLinkError, FileJob, RetryableHTTPError
from .ui import TerminalUI
from .utils import build_target_path, canonicalize_url, clean_filename


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
    album_id_match = re.match(r"^https?://[^/]+/a/([A-Za-z0-9]+)", target_url)
    album_id = album_id_match.group(1) if album_id_match else None
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
