"""
npy_to_wav.py

Convert .npy time-series files (single-channel) to WAV and optionally to MP3.
- Uses scipy.io.wavfile.write to create WAV files (int16 scaled).
- Optionally calls ffmpeg to convert WAV -> MP3 (ffmpeg must be on PATH).

Usage (PowerShell):
  python npy_to_wav.py -d "D:\ALIENS-DATASET-ECE" --recursive --sr 2048 --to-mp3 --bitrate 192k

Notes:
- Input .npy files should contain 1-D numpy arrays representing audio-like signals.
- The script normalizes each signal to int16 range before saving.
"""

import argparse
from pathlib import Path
import numpy as np
import os
import subprocess
import sys
from scipy.io import wavfile


def find_npy_files(directory: Path, recursive: bool):
    if recursive:
        return list(directory.rglob('*.npy'))
    return list(directory.glob('*.npy'))


def ffmpeg_present():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def write_wav(path: Path, arr: np.ndarray, sr: int, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return (False, 'exists')
    # ensure 1-D or 2-D (channels)
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 2:
        # transpose expected shape (n_samples, n_channels)
        a = a.T
    if a.ndim > 2:
        return (False, 'unsupported array shape')
    # normalize to int16
    maxabs = np.max(np.abs(a))
    if maxabs < 1e-12:
        scaled = np.zeros_like(a, dtype=np.int16)
    else:
        norm = a / maxabs
        scaled = (norm * 32767.0).astype(np.int16)
    try:
        wavfile.write(str(path), int(sr), scaled)
        return (True, None)
    except Exception as e:
        return (False, str(e))


def convert_wav_to_mp3_ffmpeg(wav_path: Path, mp3_path: Path, bitrate: str, overwrite: bool):
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if mp3_path.exists() and not overwrite:
        return (False, 'exists')
    cmd = [
        'ffmpeg', '-y' if overwrite else '-n', '-hide_banner', '-loglevel', 'error',
        '-i', str(wav_path),
        '-codec:a', 'libmp3lame', '-b:a', bitrate,
        str(mp3_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        return (True, None)
    except subprocess.CalledProcessError as e:
        return (False, str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', default='.', help='Directory to scan for .npy files')
    p.add_argument('-o', '--out', default=None, help='Output directory root (if not set, writes WAV next to .npy)')
    p.add_argument('-r', '--recursive', action='store_true', help='Scan directories recursively')
    p.add_argument('--sr', type=int, default=2048, help='Sampling rate for WAV files')
    p.add_argument('--to-mp3', action='store_true', help='Also convert generated WAV files to MP3 using ffmpeg')
    p.add_argument('--bitrate', default='192k', help='MP3 bitrate (e.g., 128k, 192k)')
    p.add_argument('--skip-existing', action='store_true', help='Skip when target files already exist')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = p.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(2)

    if args.to_mp3 and not ffmpeg_present():
        print('ffmpeg not found on PATH; MP3 conversion will not be available.')
        print('Install ffmpeg or rerun without --to-mp3')
        sys.exit(3)

    npys = find_npy_files(base, args.recursive)
    if not npys:
        print(f'No .npy files found in {base} (recursive={args.recursive})')
        return

    converted_wav = 0
    skipped_wav = 0
    failed_wav = 0
    converted_mp3 = 0
    skipped_mp3 = 0
    failed_mp3 = 0

    for npy in npys:
        try:
            arr = np.load(str(npy))
        except Exception as e:
            print(f"FAILED load: {npy}: {e}")
            failed_wav += 1
            continue

        if args.out:
            rel = npy.relative_to(base)
            wav_out_dir = Path(args.out) / rel.parent
            wav_path = wav_out_dir / (npy.stem + '.wav')
        else:
            wav_path = npy.with_suffix('.wav')

        ok, err = write_wav(wav_path, arr, args.sr, overwrite=args.overwrite)
        if ok:
            converted_wav += 1
            print(f"WAV: {npy} -> {wav_path}")
        else:
            if err == 'exists':
                skipped_wav += 1
                print(f"SKIP WAV exists: {wav_path}")
            else:
                failed_wav += 1
                print(f"FAILED WAV: {wav_path}: {err}")
            # still may attempt mp3 if to_mp3 and wav exists

        if args.to_mp3:
            if args.out:
                rel = npy.relative_to(base)
                mp3_out_dir = Path(args.out) / rel.parent
                mp3_path = mp3_out_dir / (npy.stem + '.mp3')
            else:
                mp3_path = npy.with_suffix('.mp3')

            # ensure wav_path exists before conversion
            if not wav_path.exists():
                print(f"Cannot convert to MP3; WAV missing: {wav_path}")
                failed_mp3 += 1
                continue

            ok2, err2 = convert_wav_to_mp3_ffmpeg(wav_path, mp3_path, args.bitrate, overwrite=args.overwrite)
            if ok2:
                converted_mp3 += 1
                print(f"MP3: {wav_path} -> {mp3_path}")
            else:
                if err2 == 'exists':
                    skipped_mp3 += 1
                    print(f"SKIP MP3 exists: {mp3_path}")
                else:
                    failed_mp3 += 1
                    print(f"FAILED MP3: {mp3_path}: {err2}")

    print('Done.')
    print(f'WAV converted: {converted_wav}, skipped: {skipped_wav}, failed: {failed_wav}')
    if args.to_mp3:
        print(f'MP3 converted: {converted_mp3}, skipped: {skipped_mp3}, failed: {failed_mp3}')


if __name__ == '__main__':
    main()
