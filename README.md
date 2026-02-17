# Bunkr Downloader (Python)

> Use only for files you have permission to download.
> This project was written entirely with AI based on user instructions.

## About

- Python script for downloading files from Bunkr album/file links.
- Supports single URL and `urls.txt` batch mode (1 link per line).
- Code in this repository was generated using AI (OpenAI Codex).

## Install

```bash
pip install requests
```

## Usage

```bash
python bunkr_downloader.py "https://bunkr.cr/a/aaIQGCuR"
```

## Batch (1 link per line)

Create `urls.txt`:

```text
https://bunkr.cr/a/aaIQGCuR
https://bunkr.cr/a/raCVNfhf
```

Run:

```bash
python bunkr_downloader.py urls.txt
```

Output:
- Album URL: `downloads/<album_id>/...`
- Example: `downloads/aaIQGCuR/your_file.mp4`
