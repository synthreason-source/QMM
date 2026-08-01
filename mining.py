import hashlib
import struct
import math
import time
import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM PoW MINER  —  Procedural Memory-Efficient Subspace Grover
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_HEADER = "First quantum sha256 by George W 1-8-2026"

# ── REGISTER SIZING ────────────────────────────────────────────────────────────
N_BITS       = 32
DIFF_BITS    = 32         # Keep <= 16-20 for fast CPU execution
MARGIN_BITS  = 4

# Cap FREE_BITS so FIXED_BITS never becomes negative
FREE_BITS    = min(DIFF_BITS + MARGIN_BITS, N_BITS)
FIXED_BITS   = N_BITS - FREE_BITS
FIXED_SUFFIX = 0

LAMBDA       = (2 ** FREE_BITS) / (2 ** DIFF_BITS)

CANDIDATE_SET = None

def index_to_raw(x: int) -> int:
    return x

def index_to_nonce(x: int) -> int:
    return (index_to_raw(x) << FIXED_BITS) | FIXED_SUFFIX

# ── SHA-256 IMPLEMENTATION ────────────────────────────────────────────────────
def pow_hash_hex(nonce: int) -> str:
    return hashlib.sha256(f"{BLOCK_HEADER}|nonce={nonce}".encode()).hexdigest()

def leading_zeros(h: str) -> int:
    bits = bin(int(h, 16))[2:].zfill(256)
    return len(bits) - len(bits.lstrip('0'))

def nonce_meets_difficulty(nonce: int) -> bool:
    return leading_zeros(pow_hash_hex(nonce)) >= DIFF_BITS

def oracle_function(x: int) -> bool:
    if CANDIDATE_SET is not None and x not in CANDIDATE_SET:
        return False
    return nonce_meets_difficulty(index_to_nonce(x))

def optimal_k(N, M):
    if M == 0 or M >= N: return 1
    return max(1, round(math.pi / (4 * math.asin(math.sqrt(M/N))) - 0.5))

# ── MEMORY-EFFICIENT PROCEDURAL ORACLE ─────────────────────────────────────────
def build_oracle_procedural(free_bits: int):
    """
    Evaluates states procedurally with tqdm.
    Prints the winning index, nonce, and hash as soon as each match is found.
    """
    dim = 2 ** free_bits
    marked = []
    
    # Stream states and record ONLY valid/marked indices
    for x in tqdm(range(dim), desc="  Evaluating Oracle States", unit="state", ascii=" █"):
        if oracle_function(x):
            marked.append(x)
            nonce = index_to_nonce(x)
            h = pow_hash_hex(nonce)
            # tqdm.write prevents output from breaking the progress bar rendering
            tqdm.write(f"  [+] HIT FOUND | Index: {x:<8} | Nonce: {nonce:<10} | Hash: {h}")
            
    return marked, dim

# ── 2D SUBSPACE GROVER SIMULATION (O(1) Memory) ──────────────────────────────
def run_grover_subspace(dim: int, marked_indices: list, iterations: int):
    """
    Simulates Grover's algorithm in O(1) memory by tracking 2D subspace amplitudes:
      a_m : amplitude on marked states
      a_u : amplitude on unmarked states
    """
    M = len(marked_indices)
    N = dim
    
    # Initial uniform superposition amplitude: 1 / sqrt(N)
    a_m = 1.0 / math.sqrt(N)
    a_u = 1.0 / math.sqrt(N)
    
    for _ in range(iterations):
        # 1. Oracle phase inversion on marked states
        a_m = -a_m
        
        # 2. Inversion about the mean (Diffusion)
        mean = (M * a_m + (N - M) * a_u) / N
        a_m = 2.0 * mean - a_m
        a_u = 2.0 * mean - a_u

    return a_m, a_u

# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 80)
    print(f"  PROCEDURAL GROVER SEARCH (O(M) Memory Footprint)")
    print("═" * 80)
    print(f"  Difficulty (D)      : {DIFF_BITS} leading zero bits")
    print(f"  Free bits           : {FREE_BITS} (2^{FREE_BITS} = {2**FREE_BITS:,} states)")
    print(f"  Expected hits (λ)   : {LAMBDA:.1f}")
    print("═" * 80)
    print()

    t0 = time.time()
    marked, N = build_oracle_procedural(FREE_BITS)
    M = len(marked)
    print(f"\n  ...Found {M} marked states out of {N:,} in {time.time()-t0:.2f}s")

    if M == 0:
        raise RuntimeError("No marked indices found. Increase MARGIN_BITS.")

    k = optimal_k(N, M)
    print(f"  Optimal Grover iterations (k): {k}")

    # Run subspace simulation using O(1) extra memory
    a_marked, a_unmarked = run_grover_subspace(N, marked, k)

    prob_marked = a_marked ** 2
    prob_unmarked = a_unmarked ** 2
    total_marked_prob = M * prob_marked

    print(f"\n── Quantum State Results ─────────────────────────────────────────────────")
    print(f"  Probability per marked state  : {prob_marked:.6f}")
    print(f"  Probability per unmarked state: {prob_unmarked:.6e}")
    print(f"  Total Success Probability     : {total_marked_prob * 100:.2f}%")

    # Sample from marked states based on total success probability
    rng = np.random.default_rng()
    if rng.random() < total_marked_prob:
        winner_idx = rng.choice(marked)
        winner_nonce = index_to_nonce(winner_idx)
        print(f"\n── Block Result ────────────────────────────────────────────────────────")
        print(f"  ✓ VALID BLOCK MINED!")
        print(f"  Measured Register Index : {winner_idx}")
        print(f"  Winning Nonce           : {winner_nonce}")
        print(f"  SHA-256 Hash            : {pow_hash_hex(winner_nonce)}")
    else:
        print("\n  ✗ Sampled an unmarked state.")
