import argparse
import sys
from pathlib import Path

from .core import run_target
from .state import FailedLinkLogger, SessionFactory
from .ui import TerminalUI
from .utils import parse_target_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from Bunkr album/file URLs.",
    )
    parser.add_argument(
        "input",
        help="Single URL or text file path (1 URL per line, e.g. urls.txt)",
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
