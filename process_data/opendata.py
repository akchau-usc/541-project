#!/usr/bin/env python3
"""
wav2csv.py  – Scan a folder of WAV files and save their metadata to a CSV.

Example
-------
python wav2csv.py data/Audio_Files -o metadata.csv
"""

import contextlib
import wave
from pathlib import Path
import csv
import argparse

def get_wav_info(path: Path, root: Path) -> dict:
    """Return a metadata dictionary for a single WAV file."""
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        frames      = wf.getnframes()
        sr          = wf.getframerate()
        channels    = wf.getnchannels()
        sampwidth   = wf.getsampwidth()          # bytes per sample
        duration    = frames / sr if sr else 0.0

    return {
        "file_name"       : path.name,
        "label"           : path.parent.name,    # folder name (optional, handy for datasets)
        "size_bytes"      : path.stat().st_size,
        "duration_sec"    : round(duration, 3),
        "sample_rate_Hz"  : sr,
        "channels"        : channels,
        "bit_depth"       : sampwidth * 8        # 8, 16, 24 or 32
    }

def write_csv(records: list[dict], csv_path: Path) -> None:
    """Write *records* to *csv_path*. Overwrites any existing file."""
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from all WAV files in a folder → CSV")
    parser.add_argument("directory",
                        help="Root directory to scan (recursively)")
    parser.add_argument("-o", "--output", default="wav_metadata.csv",
                        help="CSV filename to create (default: wav_metadata.csv)")
    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory or does not exist.")

    wav_files = list(root.rglob("*.wav"))
    if not wav_files:
        print(f"No WAV files found under {root}")
        return

    records = [get_wav_info(p, root) for p in wav_files]
    write_csv(records, Path(args.output))

    print(f"✓  Wrote metadata for {len(records)} file(s) → {args.output}")
    print("Preview of first three rows:")
    for r in records[:3]:
        print("  ", r)

if __name__ == "__main__":
    main()
