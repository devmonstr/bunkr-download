import os
import shutil
import sys
import threading
import time
from typing import Optional

from .utils import human_bytes


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
        self.multi_pct_step = 25
        self.multi_interval = 15.0
        self.multi_unknown_step_bytes = 300 * 1024 * 1024

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
        self._dashboard_ensure()
        if key in self.dashboard_key_to_slot:
            idx = self.dashboard_key_to_slot[key]
            if idx >= len(self.dashboard_slots):
                self.dashboard_slots.extend([""] * (idx - len(self.dashboard_slots) + 1))
            self.dashboard_slots[idx] = self._truncate(title)
            return
        used = set(self.dashboard_key_to_slot.values())
        idx = None
        for i in range(self.workers):
            if i not in used:
                idx = i
                break
        if idx is None:
            idx = 0
        self.dashboard_key_to_slot[key] = idx
        if idx >= len(self.dashboard_slots):
            self.dashboard_slots.extend([""] * (idx - len(self.dashboard_slots) + 1))
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
                    if idx >= len(self.dashboard_slots):
                        self.dashboard_slots.extend([""] * (idx - len(self.dashboard_slots) + 1))
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
                # Before any task slot is allocated, print normal logs directly.
                # This avoids cursor-control output swallowing early INFO/FAIL lines.
                if not self.dashboard_slots:
                    print(text, flush=True)
                    return
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
