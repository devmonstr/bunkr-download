# Bunkr Downloader (Python)

> Use only for files you have permission to download.

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
