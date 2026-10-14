"""
generate_alien_dataset.py

Creates a synthetic dataset (for research/testing only) using a simple evolutionary
algorithm to generate "alien-like" signatures that are spectrally distinct from a
background set of astrophysical-like noise signals.

USAGE (PowerShell):
    python generate_alien_dataset.py --out D:\ALIENS-DATASET-ECE --num_background 200 --num_alien 50

NOTES:
- This produces synthetic signals only. These are NOT real detections.
- The evolutionary algorithm and fitness function are intentionally simple and
  intended for generating varied test data, not for scientific detection.

Dependencies: numpy scipy matplotlib pandas
"""

import os
import argparse
import numpy as np
import scipy.signal as ss
import scipy.fft as fft
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import csv
import random

# ---------- Utilities -----------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_wave_numpy(path, arr, sr):
    np.save(path, np.asarray(arr, dtype=np.float32))


def save_spectrogram_png(path, arr, sr, vmin=None, vmax=None):
    f, t, Sxx = ss.spectrogram(arr, fs=sr, nperseg=256, noverlap=192)
    plt.figure(figsize=(6,3))
    plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='gouraud', vmin=vmin, vmax=vmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Background signal generator ---------------------------------

def generate_background_signal(duration_s=5.0, sr=2048, seed=None):
    """Produce a background "astrophysical" like time series: pink-ish noise,
    random chirps and occasional pulses."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    n = int(duration_s * sr)
    # pink-ish noise via filtering white noise
    white = rng.normal(0, 1, n)
    b, a = ss.butter(2, [0.001, 0.5], btype='band')
    noise = ss.lfilter(b, a, white)
    # add random transient pulses
    for _ in range(rng.randint(1, 4)):
        center = rng.randint(0, n)
        width = rng.randint(int(sr*0.01), int(sr*0.2))
        t0 = max(0, center - width//2)
        t1 = min(n, center + width//2)
        amp = rng.uniform(0.5, 4.0)
        # tukey window may be in scipy.signal.windows depending on scipy version
        try:
            window = ss.tukey(t1-t0, alpha=0.5)
        except Exception:
            try:
                window = ss.windows.tukey(t1-t0, alpha=0.5)
            except Exception:
                # fallback to simple Hanning
                window = np.hanning(max(1, t1-t0))
        noise[t0:t1] += amp * window * rng.normal(0, 1, t1-t0)
    # occasional chirp
    if rng.rand() < 0.5:
        t = np.linspace(0, duration_s, n)
        f0 = rng.uniform(5, 100)
        f1 = rng.uniform(100, 800)
        chirp = ss.chirp(t, f0=f0, t1=duration_s, f1=f1, method='quadratic')
        noise += 0.1 * chirp * rng.uniform(0.5, 2.0)
    # normalize
    noise = noise / (np.std(noise) + 1e-12)
    return noise


# ---------- Candidate (alien) signal representation ----------------------
# We'll represent each candidate as a sum of M sinusoids with Gaussian envelopes

class Candidate:
    def __init__(self, freqs, amps, phases, centers, widths, sr, duration):
        self.freqs = np.array(freqs)
        self.amps = np.array(amps)
        self.phases = np.array(phases)
        self.centers = np.array(centers)
        self.widths = np.array(widths)
        self.sr = sr
        self.duration = duration

    def synth(self):
        t = np.linspace(0, self.duration, int(self.duration*self.sr), endpoint=False)
        sig = np.zeros_like(t)
        for f,a,phi,c,w in zip(self.freqs, self.amps, self.phases, self.centers, self.widths):
            env = np.exp(-0.5 * ((t - c)/ (w + 1e-9))**2)
            sig += a * env * np.sin(2*np.pi*f*t + phi)
        # small noise and normalization
        sig = sig / (np.std(sig) + 1e-12)
        return sig

    def copy(self):
        return Candidate(np.copy(self.freqs), np.copy(self.amps), np.copy(self.phases), np.copy(self.centers), np.copy(self.widths), self.sr, self.duration)


# ---------- Evolutionary operators --------------------------------------

def random_candidate(M, sr, duration, rng):
    freqs = rng.uniform(3, sr/2*0.9, size=M)
    amps = rng.uniform(0.2, 4.0, size=M)
    phases = rng.uniform(0, 2*np.pi, size=M)
    centers = rng.uniform(0, duration, size=M)
    widths = rng.uniform(0.01*duration, 0.5*duration, size=M)
    return Candidate(freqs, amps, phases, centers, widths, sr, duration)


def crossover(a: Candidate, b: Candidate, rng):
    # single-point crossover on parameters
    child = a.copy()
    M = len(a.freqs)
    point = rng.randint(1, M)
    child.freqs[point:] = b.freqs[point:]
    child.amps[point:] = b.amps[point:]
    child.phases[point:] = b.phases[point:]
    child.centers[point:] = b.centers[point:]
    child.widths[point:] = b.widths[point:]
    return child


def mutate(candidate: Candidate, rng, freq_mut_scale=0.05, amp_mut_scale=0.2):
    c = candidate.copy()
    # mutate frequencies
    c.freqs += rng.normal(0, freq_mut_scale * c.freqs)
    c.freqs = np.clip(c.freqs, 1.0, c.sr/2*0.95)
    # mutate amplitudes
    c.amps += rng.normal(0, amp_mut_scale * np.maximum(0.1, c.amps))
    c.amps = np.clip(c.amps, 0.01, 10.0)
    # mutate phases
    c.phases += rng.normal(0, 0.2, size=c.phases.shape)
    # centers and widths
    c.centers += rng.normal(0, 0.05 * c.duration, size=c.centers.shape)
    c.centers = np.clip(c.centers, 0, c.duration)
    c.widths += rng.normal(0, 0.05 * c.duration, size=c.widths.shape)
    c.widths = np.clip(c.widths, 0.001*c.duration, 0.7*c.duration)
    return c


# ---------- Fitness function --------------------------------------------

def spectrum_distance_to_background(sig, bg_mean_spectrum, sr):
    # compute log-power spectrum and Euclidean distance to mean background
    N = len(sig)
    S = np.abs(fft.rfft(sig))
    S = S / (np.sum(S) + 1e-12)
    sm = bg_mean_spectrum
    d = np.linalg.norm(S - sm)
    return d


# ---------- Evolutionary loop ------------------------------------------

def evolve_signatures(bg_spectra_mean, sr, duration, num_output, pop_size=80, generations=60, M=6, rng_seed=None):
    rng = np.random.RandomState(rng_seed)
    population = [random_candidate(M, sr, duration, rng) for _ in range(pop_size)]
    # evaluate
    def fitness_fn(candidate):
        sig = candidate.synth()
        d = spectrum_distance_to_background(sig, bg_spectra_mean, sr)
        # penalize overly narrow or trivial signals (small variance)
        penalty = 1.0 / (np.std(sig) + 1e-9)
        return d - 0.5 * penalty

    for gen in range(generations):
        scores = [fitness_fn(c) for c in population]
        # select top fraction
        idx = np.argsort(scores)[::-1]
        topk = int(0.2 * pop_size)
        elites = [population[i] for i in idx[:topk]]
        # produce offspring
        newpop = elites.copy()
        while len(newpop) < pop_size:
            a = rng.choice(elites)
            b = rng.choice(population)
            child = crossover(a, b, rng)
            if rng.rand() < 0.7:
                child = mutate(child, rng)
            newpop.append(child)
        population = newpop
        # optionally print progress
        if gen % 10 == 0 or gen == generations-1:
            best_score = scores[idx[0]]
            print(f"Evolve gen {gen}/{generations} best {best_score:.4f}")
    # final ranking
    final_scores = [fitness_fn(c) for c in population]
    idx = np.argsort(final_scores)[::-1]
    best = [population[i] for i in idx[:num_output]]
    return best, [final_scores[i] for i in idx[:num_output]]


# ---------- Main dataset creation --------------------------------------

def main(out_dir, num_background=200, num_alien=50, duration=5.0, sr=2048, rng_seed=42):
    ensure_dir(out_dir)
    bg_dir = os.path.join(out_dir, 'background')
    alien_dir = os.path.join(out_dir, 'alien')
    ensure_dir(bg_dir)
    ensure_dir(alien_dir)
    rng = np.random.RandomState(rng_seed)

    print("Generating background signals...")
    bg_spectra = []
    for i in range(num_background):
        sig = generate_background_signal(duration_s=duration, sr=sr, seed=rng.randint(0,2**31-1))
        save_wave_numpy(os.path.join(bg_dir, f'bg_{i:04d}.npy'), sig, sr)
        save_spectrogram_png(os.path.join(bg_dir, f'bg_{i:04d}.png'), sig, sr)
        # spectrum normalized
        S = np.abs(fft.rfft(sig))
        S = S / (np.sum(S) + 1e-12)
        bg_spectra.append(S)

    bg_mean_spectrum = np.mean(np.stack(bg_spectra, axis=0), axis=0)
    print("Background generation complete. Evolving alien signatures...")

    # evolve to create num_alien signatures in batches
    batch = min(10, num_alien)
    generated = []
    scores_all = []
    to_generate = num_alien
    batch_id = 0
    while to_generate > 0:
        n = min(batch, to_generate)
        best, scores = evolve_signatures(bg_mean_spectrum, sr, duration, n, pop_size=80, generations=60, M=6, rng_seed=rng.randint(0,2**31-1))
        for j, (cand, sc) in enumerate(zip(best, scores)):
            idx = len(generated)
            sig = cand.synth()
            fname = f'alien_{idx:04d}'
            save_wave_numpy(os.path.join(alien_dir, fname + '.npy'), sig, sr)
            save_spectrogram_png(os.path.join(alien_dir, fname + '.png'), sig, sr)
            generated.append(fname)
            scores_all.append(float(sc))
        to_generate -= n
        batch_id += 1

    # write metadata
    meta_path = os.path.join(out_dir, 'metadata.csv')
    with open(meta_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'type', 'file_npy', 'file_png', 'score', 'created_at'])
        # background entries
        for i in range(num_background):
            writer.writerow([f'bg_{i:04d}', 'background', os.path.join('background', f'bg_{i:04d}.npy'), os.path.join('background', f'bg_{i:04d}.png'), '', datetime.utcnow().isoformat() + 'Z'])
        for i, (name, sc) in enumerate(zip(generated, scores_all)):
            writer.writerow([name, 'alien', os.path.join('alien', name + '.npy'), os.path.join('alien', name + '.png'), f'{sc:.6f}', datetime.utcnow().isoformat() + 'Z'])

    print(f"Dataset generation complete: {out_dir}")
    print(f"Background: {num_background}, Alien (synthetic): {num_alien}")


# ---------- CLI ---------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic alien-signature dataset using evolutionary search')
    parser.add_argument('--out', '-o', default='D:/ALIENS-DATASET-ECE', help='Output directory')
    parser.add_argument('--num_background', type=int, default=200)
    parser.add_argument('--num_alien', type=int, default=50)
    parser.add_argument('--duration', type=float, default=5.0, help='Signal duration in seconds')
    parser.add_argument('--sr', type=int, default=2048, help='Sampling rate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.out, num_background=args.num_background, num_alien=args.num_alien, duration=args.duration, sr=args.sr, rng_seed=args.seed)
