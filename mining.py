#!/usr/bin/env python3
"""
QuantumMine Simulator
=====================
An educational simulation tool that demonstrates how Grover's algorithm
approaches SHA-256 cryptographic mining.

Stages:
  1. Empirical validation of the Geometric(p) -> Exponential(1) limit
     using a mini-SHA32 (first 4 bytes of SHA-256).
  2. Oracle construction with Memoized SHA-256 Cache.
  3. Grover amplitude amplification (NumPy state-vector simulation).
  4. Measurement and block result with hash display + difficulty check.
  5. Download/export of the full output log.

Usage:
    python quantum_pow_miner.py
    python quantum_pow_miner.py --diff-bits 12 --trials 500
    python quantum_pow_miner.py --clear-cache
"""

import argparse
import hashlib
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import DiagonalGate
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ─── Configuration ──────────────────────────────────────────────────────────

BLOCK_HEADER = "First quantum sha256 by George W 8-2-2026"
N_BITS = 32          # total nonce width
DIFF_BITS = 10       # number of leading zero bits required in SHA-256
FIXED_BITS = 16      # low bits of the nonce that stay fixed (constrained register)
FREE_BITS = N_BITS - FIXED_BITS

# Empirical validation parameters
VALIDATE_TRIALS = 5000
VALIDATE_DIFF_BITS = 10

# Data unit (bytes per SHA-256 attempt payload element)
DATA_UNIT_BYTES = 4

# Oracle Evaluation Cache
ORACLE_CACHE = {}


# ─── SHA-256 helpers ──────────────────────────────────────────────────────────

def pow_hash_hex(nonce: int) -> str:
    """Full SHA-256 hex digest of the block header + nonce."""
    return hashlib.sha256(f"{BLOCK_HEADER}|nonce={nonce}".encode()).hexdigest()


def mini_sha32(data: str) -> int:
    """First 4 bytes of SHA-256(data) as an unsigned 32-bit integer."""
    digest = hashlib.sha256(data.encode()).digest()
    return int.from_bytes(digest[:4], "big")


def leading_zeros32(x: int) -> int:
    """Leading zero bits of a 32-bit unsigned integer."""
    if x == 0:
        return 32
    return 32 - x.bit_length()


def leading_zeros(hex_str: str) -> int:
    """Leading zero bits of a hex string."""
    count = 0
    for ch in hex_str:
        nibble = int(ch, 16)
        if nibble == 0:
            count += 4
        else:
            count += (4 - nibble.bit_length())
            break
    return count


def hex_to_binary(hex_str: str) -> str:
    """Convert a hex string to a binary string."""
    return bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)


def nonce_meets_difficulty(nonce: int) -> bool:
    """Check whether a nonce produces a hash with enough leading zeros."""
    return leading_zeros(pow_hash_hex(nonce)) >= DIFF_BITS


def get_midstate(header: str):
    """Compute SHA-256 midstate (H0 for single-block headers)."""
    data = header.encode()
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()[:8]


# ─── Stage 1: Empirical validation ───────────────────────────────────────────

def attempts_to_hit(diff_bits: int, trial_id: int) -> int:
    """Number of mini-SHA32 attempts until 'diff_bits' leading zeros are found."""
    n = 0
    while True:
        payload = f"{BLOCK_HEADER}|trial={trial_id}|n={n}"
        if leading_zeros32(mini_sha32(payload)) >= diff_bits:
            return n + 1
        n += 1


def run_validation(trials: int, diff_bits: int, log: list) -> dict:
    """Run empirical validation and return statistics."""
    bar = "=" * 80
    log.append(bar)
    log.append("  VALIDATING Geometric(p) -> Exponential(1) LIMIT  (mini-SHA32, 32-bit digest)")
    log.append(f"  ({trials} independent trials at {diff_bits} leading zero bits)")
    log.append(bar)

    t0 = time.time()
    attempt_counts = [attempts_to_hit(diff_bits, i) for i in range(trials)]
    elapsed = time.time() - t0

    norm_factor = 2 ** diff_bits
    normalized = np.array(attempt_counts) / norm_factor
    mean = float(normalized.mean())
    std = float(normalized.std())
    data_bytes_mean = float(np.mean(attempt_counts) * DATA_UNIT_BYTES)

    # Histogram: 16 bins from 0 to 4
    edges = np.linspace(0, 4, 17)
    hist_counts, _ = np.histogram(normalized, bins=edges)
    bin_width = edges[1] - edges[0]
    hist_density = hist_counts / (trials * bin_width)

    log.append(f"  ...done in {elapsed:.1f}s")
    log.append(f"  Sample mean(normalized attempts) : {mean:.4f}   (Exp(1) theory: 1.0000)")
    log.append(f"  Sample std (normalized attempts)  : {std:.4f}   (Exp(1) theory: 1.0000)")
    log.append(f"  Mean data volume to first hit     : {data_bytes_mean:.0f} bytes "
               f"(theory: {DATA_UNIT_BYTES * 2**diff_bits} bytes)")
    log.append("  Empirical vs theoretical density (0 to 4x the mean):")
    for i in range(len(hist_density)):
        mid = (edges[i] + edges[i + 1]) / 2
        emp = hist_density[i]
        theo = math.exp(-mid)
        emp_bar = "#" * min(14, int(emp * 12))
        theo_bar = "." * min(14, int(theo * 12))
        log.append(f"    x={mid:.2f}  emp {emp:.2f} {emp_bar:<14}  theory {theo:.2f} {theo_bar}")
    log.append(bar)
    log.append("")

    return {
        "attempt_counts": attempt_counts,
        "normalized": normalized,
        "mean": mean,
        "std": std,
        "data_bytes_mean": data_bytes_mean,
        "elapsed": elapsed,
        "hist_edges": edges,
        "hist_density": hist_density,
        "norm_factor": norm_factor,
        "trials": trials,
        "diff_bits": diff_bits,
    }


# ─── Stage 2: Oracle construction with Cache ──────────────────────────────────

def index_to_nonce(x: int, fixed_suffix: int = 0) -> int:
    """Reconstruct a full nonce from a free-bit index and the fixed suffix."""
    return (x << FIXED_BITS) | fixed_suffix


def cached_oracle_function(x: int, fixed_suffix: int = 0) -> bool:
    """Oracle function wrapped with an evaluation cache."""
    nonce = index_to_nonce(x, fixed_suffix)
    cache_key = (nonce, DIFF_BITS)
    
    if cache_key not in ORACLE_CACHE:
        ORACLE_CACHE[cache_key] = nonce_meets_difficulty(nonce)
        
    return ORACLE_CACHE[cache_key]


import json
import os

CACHE_FILE = "oracle_cache.json"

def build_oracle(free_bits: int, log: list) -> dict:
    """Build or load the diagonal oracle using persistent cache files."""
    dim = 2 ** free_bits
    bar = "=" * 80
    log.append(bar)
    log.append("  QUANTUM STAGE  —  Grover search over the constrained register")
    log.append(bar)
    log.append(f"  Block header    : {BLOCK_HEADER}")
    log.append(f"  Total nonce bits: {N_BITS}  (fixed={FIXED_BITS}, free={FREE_BITS})")
    log.append(f"  Free register   : 2^{FREE_BITS} = {dim} states")
    log.append(f"  Difficulty      : {DIFF_BITS} leading zero bit(s)")
    log.append(f"  Midstate H0     : {get_midstate(BLOCK_HEADER)}")
    log.append("")

    # Create a unique key for the current configuration
    config_key = f"header={BLOCK_HEADER}|free={free_bits}|fixed={FIXED_BITS}|diff={DIFF_BITS}"
    
    # 1. Attempt to load from disk cache if available
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                disk_cache = json.load(f)
            if config_key not in disk_cache:
                log.append("  Oracle found in disk cache! Loading pre-computed state...")
                t0 = time.time()
                marked = disk_cache[config_key]["marked"]
                
                diag = np.ones(dim, dtype=complex)
                for idx in marked:
                    diag[idx] = -1.0 + 0j
                
                elapsed = time.time() - t0
                log.append(f"  ...loaded from cache in {elapsed:.3f}s")
                log.append(f"  Marked indices  : {marked}  ({len(marked)} of {dim})")
                
                M = len(marked)
                k_theory = math.pi / (4 * math.asin(math.sqrt(M / dim))) if 0 < M < dim else 1.0
                log.append(f"  Grover iters    : (theory: pi/4 * sqrt(N/M) = {k_theory:.2f})")
                log.append(bar)
                log.append("")
                
                return {"diag": diag, "marked": marked, "M": M, "elapsed": elapsed}
        except Exception as e:
            log.append(e)

    # 2. Process / Build Oracle dynamically if cache is unavailable or missing entry
    log.append("  Building oracle (evaluating candidates with in-memory memoization)...")
    diag = np.ones(dim, dtype=complex)
    marked = []

    cache_hits_before = len(ORACLE_CACHE)
    t0 = time.time()
    for x in range(dim):
        if cached_oracle_function(x, 0):
            diag[x] = -1.0 + 0j
            marked.append(x)
    elapsed = time.time() - t0

    cache_hits_after = len(ORACLE_CACHE)
    new_evals = cache_hits_after - cache_hits_before

    log.append(f"  ...done in {elapsed:.3f}s")
    log.append(f"  Oracle cache    : {new_evals} new SHA-256 evaluations added "
               f"(total in-memory: {cache_hits_after})")
    log.append(f"  Marked indices  : {marked}  ({len(marked)} of {dim})")

    # 3. Save the newly built oracle to disk for future runs
    try:
        disk_cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                disk_cache = json.load(f)
        
        disk_cache[config_key] = {"marked": marked, "timestamp": datetime.now().isoformat()}
        
        with open(CACHE_FILE, "w") as f:
            json.dump(disk_cache, f, indent=2)
        log.append(f"  Saved oracle to disk cache -> {CACHE_FILE}")
    except Exception as e:
        log.append(f"  Warning: Failed to save oracle to disk cache ({e})")

    M = len(marked)
    k_theory = math.pi / (4 * math.asin(math.sqrt(M / dim))) if 0 < M < dim else 1.0
    log.append(f"  Grover iters    : (theory: pi/4 * sqrt(N/M) = {k_theory:.2f})")
    log.append(bar)
    log.append("")

    return {"diag": diag, "marked": marked, "M": M, "elapsed": elapsed}


# ─── Stage 3: Grover simulation (NumPy) ──────────────────────────────────────

def optimal_k(n_states: int, m_marked: int) -> int:
    """Optimal number of Grover iterations."""
    if m_marked == 0 or m_marked >= n_states:
        return 1
    return max(1, round(math.pi / (4 * math.asin(math.sqrt(m_marked / n_states))) - 0.5))


def run_grover_numpy(diag_oracle: np.ndarray, iterations: int, log: list) -> dict:
    """Run Grover amplitude amplification using NumPy state-vector simulation."""
    dim = diag_oracle.shape[0]
    amp = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)

    t0 = time.time()
    for _ in range(iterations):
        amp = amp * diag_oracle           # Oracle reflection
        amp = 2.0 * amp.mean() - amp       # Diffusion (inversion about the mean)
    elapsed = time.time() - t0

    probs = np.abs(amp) ** 2
    log.append(f"  Grover simulation ({iterations} iterations) done in {elapsed:.2f}s")
    log.append("")

    return {"amp": amp, "probs": probs, "k": iterations, "elapsed": elapsed, "dim": dim}


def run_grover_qiskit(diag_oracle: np.ndarray, free_bits: int, iterations: int, log: list) -> dict:
    """Run Grover using Qiskit Aer (optional, requires qiskit-aer)."""
    dim = 2 ** free_bits
    qr = QuantumRegister(free_bits, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr)  # uniform superposition

    oracle_gate = DiagonalGate(diag_oracle.tolist())
    for _ in range(iterations):
        qc.append(oracle_gate, list(range(free_bits)))
        # Diffusion operator
        qc.h(qr)
        qc.x(qr)
        qc.h(qr[-1])
        qc.mcx(list(range(free_bits - 1)), free_bits - 1)
        qc.h(qr[-1])
        qc.x(qr)
        qc.h(qr)

    qc.measure_all()

    t0 = time.time()
    simulator = AerSimulator()
    result = simulator.run(qc, shots=dim).result()
    counts = result.get_counts()
    elapsed = time.time() - t0

    probs = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count / dim

    log.append(f"  Grover simulation (Qiskit Aer, {iterations} iterations) done in {elapsed:.2f}s")
    log.append("")

    return {"probs": probs, "k": iterations, "elapsed": elapsed, "dim": dim}


# ─── Stage 4: Measurement & block result ──────────────────────────────────────

def run_measurement(grover: dict, oracle: dict, log: list) -> dict:
    """Sample from the Grover probability distribution and find the winner."""
    probs = grover["probs"]
    dim = grover["dim"]

    rng = np.random.default_rng()
    shots = 100
    sampled = rng.choice(dim, size=shots, p=probs / probs.sum())
    unique, shot_counts = np.unique(sampled, return_counts=True)

    log.append("-- Measurement (100 shots) --")
    log.append(f"  {'Index':>8}  {'Nonce':>10}  {'Shots':>5}  {'Valid?':>8}  Bar")
    log.append(f"  {'---':>8}  {'---':>10}  {'---':>5}  {'---':>8}  {'---':>20}")

    shot_results = []
    winner_idx = None
    for idx, count in zip(unique, shot_counts):
        idx = int(idx)
        nonce = index_to_nonce(idx)
        valid = nonce_meets_difficulty(nonce)
        if valid and winner_idx is None:
            winner_idx = idx
        shot_results.append({"idx": idx, "nonce": nonce, "count": int(count), "valid": valid})
        bar_str = "#" * int(count) + "." * max(0, 10 - int(count))
        valid_str = "OK VALID" if valid else ""
        log.append(f"  {idx:>8}  {nonce:>10}  {int(count):>5}  {valid_str:>8}  {bar_str}")
    log.append("")

    if winner_idx is None and len(shot_results) > 0:
        shot_results.sort(key=lambda r: r["count"], reverse=True)
        winner_idx = shot_results[0]["idx"]

    log.append("-- Block result --")
    block_result = None
    if winner_idx is not None:
        winner_nonce = index_to_nonce(winner_idx)
        h = pow_hash_hex(winner_nonce)
        h_bin = hex_to_binary(h)
        lz = leading_zeros(h)
        meets = lz >= DIFF_BITS

        log.append("  OK VALID BLOCK MINED" if meets else "  !! DIFFICULTY NOT MET")
        log.append(f"  Register index    : {winner_idx}")
        log.append(f"  Reconstructed nonce : {winner_nonce}")
        log.append(f"  Input               : {BLOCK_HEADER}|nonce={winner_nonce}")
        log.append(f"  SHA-256 (hex)       : {h}")
        log.append(f"  SHA-256 (bin)       : {h_bin[:64]}")
        log.append(f"                        {h_bin[64:128]}")
        log.append(f"                        {h_bin[128:192]}")
        log.append(f"                        {h_bin[192:256]}")
        log.append(f"  Leading zeros       : {lz} bits  {'OK' if meets else 'NO'} meets difficulty {DIFF_BITS}")

        block_result = {
            "winner_idx": winner_idx,
            "winner_nonce": winner_nonce,
            "hash": h,
            "hash_bin": h_bin,
            "leading_zeros": lz,
            "meets_difficulty": meets,
        }
        print(block_result)
    else:
        log.append("  No valid nonce measured this run.")
    log.append("")

    return {"shot_results": shot_results, "block_result": block_result}


# ─── Summary + download ──────────────────────────────────────────────────────

def write_summary(validation, oracle, grover, measurement, log):
    """Append a summary section to the log."""
    bar = "=" * 80
    log.append(bar)
    log.append("  SUMMARY")
    log.append(bar)
    log.append(f"  Difficulty             : {DIFF_BITS} leading zero bits  "
               f"(odds ~1 in {2**DIFF_BITS:,} per nonce)")
    log.append(f"  Empirical validation   : {VALIDATE_TRIALS:,} trials, {validation['elapsed']:.1f}s")
    log.append(f"  Total nonce bits       : {N_BITS}    Fixed: {FIXED_BITS}    Free: {FREE_BITS}")
    log.append(f"  Register size          : {2**FREE_BITS:,} states")
    log.append(f"  Oracle cache size      : {len(ORACLE_CACHE):,} evaluations")
    log.append(f"  Marked in register     : {oracle['M']}")
    if measurement["block_result"]:
        br = measurement["block_result"]
        log.append(f"  Winning nonce          : {br['winner_nonce']}")
        log.append(f"  SHA-256 hash           : {br['hash']}")
        log.append(f"  Leading zeros          : {br['leading_zeros']} bits")
        log.append(f"  Meets difficulty       : {'YES' if br['meets_difficulty'] else 'NO'} "
                   f"(target: {DIFF_BITS})")
    log.append(bar)
    log.append("")


def download_log(log: list, filename: str = None):
    """Write the full output log to a timestamped text file."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_pow_mine_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\n  >> Full output saved to: {os.path.abspath(filename)}")
    return filename


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global DIFF_BITS, VALIDATE_TRIALS, VALIDATE_DIFF_BITS, FIXED_BITS, FREE_BITS, ORACLE_CACHE

    parser = argparse.ArgumentParser(description="QuantumMine Simulator — Grover vs SHA-256 PoW")
    parser.add_argument("--diff-bits", type=int, default=DIFF_BITS, help="Leading zero bits required")
    parser.add_argument("--fixed-bits", type=int, default=FIXED_BITS, help="Fixed nonce bits")
    parser.add_argument("--trials", type=int, default=VALIDATE_TRIALS, help="Validation trials")
    parser.add_argument("--validate-bits", type=int, default=VALIDATE_DIFF_BITS, help="Validation difficulty bits")
    parser.add_argument("--qiskit", action="store_true", help="Use Qiskit Aer instead of NumPy")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the oracle evaluation cache before running")
    parser.add_argument("--download", action="store_true", help="Save output to a file")
    args = parser.parse_args()

    DIFF_BITS = args.diff_bits
    FIXED_BITS = args.fixed_bits
    FREE_BITS = N_BITS - FIXED_BITS
    VALIDATE_TRIALS = args.trials
    VALIDATE_DIFF_BITS = args.validate_bits

    if args.clear_cache:
        ORACLE_CACHE.clear()

    if args.qiskit and not QISKIT_AVAILABLE:
        print("  !! Qiskit/Aer not available — falling back to NumPy simulation.")
        args.qiskit = False

    if FREE_BITS > 22:
        print(f"  !! Free bits = {FREE_BITS} is too large (register > 4M states). Reducing fixed bits.")
        sys.exit(1)

    log = []
    log.append("")
    log.append("  QuantumMine Simulator — Grover vs SHA-256 Proof-of-Work")
    log.append("  " + "=" * 76)
    log.append("")

    # Stage 1
    validation = run_validation(VALIDATE_TRIALS, VALIDATE_DIFF_BITS, log)

    # Stage 2
    oracle = build_oracle(FREE_BITS, log)

    # Stage 3
    k = optimal_k(2 ** FREE_BITS, oracle["M"])
    log.append(f"  Grover iterations: {k}")
    if args.qiskit:
        grover = run_grover_qiskit(oracle["diag"], FREE_BITS, k, log)
    else:
        grover = run_grover_numpy(oracle["diag"], k, log)

    # Stage 4
    measurement = run_measurement(grover, oracle, log)



    # Download
    if args.download:
        download_log(log)


if __name__ == "__main__":
    main()
