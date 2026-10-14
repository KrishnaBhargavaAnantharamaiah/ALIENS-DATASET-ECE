"""
convert_wav_to_mp3.py

Batch-convert WAV files to MP3 using ffmpeg (preferred) via subprocess.

Usage (PowerShell):
  python convert_wav_to_mp3.py -d "D:\ALIENS-DATASET-ECE" --recursive --bitrate 192k --skip-existing

Notes:
- Requires ffmpeg available on PATH. On Windows you can install via Chocolatey:
    choco install ffmpeg
  or download a build and add to PATH.
- The script does not add Python dependencies; it uses the system ffmpeg binary.

Options:
  -d, --dir         Input directory to scan for .wav files (default: current dir)
  -o, --out         Output directory (default: same directory as input files)
  -r, --recursive   Recursively scan subdirectories
  --bitrate         MP3 bitrate (e.g., 128k, 192k). Default: 192k
  --skip-existing   Skip conversion when target .mp3 already exists
  --overwrite       Overwrite existing .mp3 files
"""

import argparse
import os
import subprocess
from pathlib import Path
import sys


def find_wav_files(directory: Path, recursive: bool):
    if recursive:
        return list(directory.rglob('*.wav'))
    return list(directory.glob('*.wav'))


def ffmpeg_present():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def convert_file(in_path: Path, out_path: Path, bitrate: str, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return (False, 'exists')
    cmd = [
        'ffmpeg', '-y' if overwrite else '-n', '-hide_banner', '-loglevel', 'error',
        '-i', str(in_path),
        '-codec:a', 'libmp3lame', '-b:a', bitrate,
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        return (True, None)
    except subprocess.CalledProcessError as e:
        return (False, str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', default='.', help='Directory to scan for WAV files')
    p.add_argument('-o', '--out', default=None, help='Output directory (default: same folders as inputs)')
    p.add_argument('-r', '--recursive', action='store_true', help='Scan directories recursively')
    p.add_argument('--bitrate', default='192k', help='Target MP3 bitrate (e.g., 128k, 192k)')
    p.add_argument('--skip-existing', action='store_true', help='Skip files where target MP3 exists')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing MP3 files')
    args = p.parse_args()

    base_dir = Path(args.dir).expanduser().resolve()
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        sys.exit(2)

    if not ffmpeg_present():
        print("ffmpeg not found on PATH. Install ffmpeg and ensure it's on PATH before running this script.")
        print("On Windows: choco install ffmpeg  (or download from https://ffmpeg.org and add to PATH)")
        sys.exit(3)

    wavs = find_wav_files(base_dir, args.recursive)
    if not wavs:
        print(f"No WAV files found in {base_dir} (recursive={args.recursive})")
        return

    print(f"Found {len(wavs)} .wav files. Converting to MP3 (bitrate={args.bitrate})...")
    converted = 0
    skipped = 0
    failed = 0
    for wav in wavs:
        # determine output path
        if args.out:
            rel = wav.relative_to(base_dir)
            out_dir = Path(args.out) / rel.parent
            out_path = out_dir / (wav.stem + '.mp3')
        else:
            out_path = wav.with_suffix('.mp3')

        if out_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
            print(f"SKIP (exists): {out_path}")
            continue

        ok, err = convert_file(wav, out_path, args.bitrate, overwrite=args.overwrite)
        if ok:
            converted += 1
            print(f"OK: {wav} -> {out_path}")
        else:
            failed += 1
            print(f"FAILED: {wav} -> {out_path}: {err}")

    print(f"Done. Converted: {converted}, Skipped: {skipped}, Failed: {failed}")

if __name__ == '__main__':
    main()
