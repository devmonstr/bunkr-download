from dataclasses import dataclass
from typing import Optional
import re


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
