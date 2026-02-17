import threading
import time
from pathlib import Path
from typing import Optional

import requests

from .types import USER_AGENT


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
