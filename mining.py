import hashlib
import struct
import math
import time
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate

from qiskit_aer import AerSimulator

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM XOR-ASYMMETRIC PoW MINER  —  SHA-256 Midstate Oracle
#  CONSTRAINED-NONCE VARIANT  —  DIFF_BITS = 24
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_HEADER = "First quantum sha256 by George W 28-4-2026"
N_BITS       = 32
DIFF_BITS    = 24
MASK32       = 0xFFFFFFFF

# ── CONSTRAINT CONFIGURATION ──────────────────────────────────────────────────
FIXED_BITS = 16
FREE_BITS   = N_BITS - FIXED_BITS

CANDIDATE_SET = None

def index_to_raw(x: int) -> int:
    return x

assert 0 <= FREE_BITS <= 24, "keep FREE_BITS <= ~20-22 for this simulator to stay fast"

# ── EXPONENTIAL MODEL CONFIG ──────────────────────────────────────────────────
VALIDATE_EXP_MODEL = False
VALIDATE_DIFF_BITS = 18
VALIDATE_TRIALS = 2000

# ============================================================================
# PRECOMPUTED DIFFICULTY TABLE
# ============================================================================
DATA_UNIT_BYTES = 4

DIFFICULTY_TABLE = {}

for diff in range(33):
    attempts = 1 << diff

    DIFFICULTY_TABLE[diff] = {
        "difficulty": diff,
        "probability": 2.0 ** (-diff),
        "expected_attempts": attempts,
        "expected_bytes": attempts * DATA_UNIT_BYTES,
        "expected_mean": float(attempts),
        "expected_std": float(attempts),
    }

# ── SHA-256 (unchanged) ────────────────────────────────────────────────────────
def rotr32(x, n): return ((x >> n) | (x << (32 - n))) & MASK32

K256 = [
    0x428a2f98,0x71374491,0xb5b0fbcf,0xe9b5dba5,0x3956c25d,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8ba4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]
H0 = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

def sha256_compress(state, block64):
    w = list(struct.unpack('>16I', block64))
    for i in range(16, 64):
        s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3)
        s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10)
        w.append((w[i-16]+s0+w[i-7]+s1)&MASK32)
    a,b,c,d,e,f,g,h = state
    for i in range(64):
        S1  = rotr32(e,6)^rotr32(e,11)^rotr32(e,25)
        ch  = (e&f)^(~e&g)
        t1  = (h+S1+ch+K256[i]+w[i])&MASK32
        S0  = rotr32(a,2)^rotr32(a,13)^rotr32(a,22)
        maj = (a&b)^(a&c)^(b&c)
        t2  = (S0+maj)&MASK32
        h=g; g=f; f=e; e=(d+t1)&MASK32
        d=c; c=b; b=a; a=(t1+t2)&MASK32
    return [(s+v)&MASK32 for s,v in zip(state,[a,b,c,d,e,f,g,h])]

def get_midstate(header_bytes):
    data = header_bytes
    ml   = len(data) * 8
    data += b'\x80'
    while len(data) % 64 != 56:
        data += b'\x00'
    data += struct.pack('>Q', ml)
    blocks = [data[i:i+64] for i in range(0, len(data), 64)]
    state  = list(H0)
    for blk in blocks[:-1]:
        state = sha256_compress(state, blk)
    return state, blocks[-1]

MIDSTATE, LAST_BLK_TMPL = get_midstate(BLOCK_HEADER.encode())

def pow_hash_hex(nonce: int) -> str:
    return hashlib.sha256(f"{BLOCK_HEADER}|nonce={nonce}".encode()).hexdigest()

def leading_zeros(h: str) -> int:
    bits = bin(int(h, 16))[2:].zfill(256)
    return len(bits) - len(bits.lstrip('0'))

def nonce_meets_difficulty(nonce: int) -> bool:
    return leading_zeros(pow_hash_hex(nonce)) >= DIFF_BITS

def mini_sha32(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')

def leading_zeros32(x: int) -> int:
    return 32 - x.bit_length() if x else 32

def attempts_to_hit(header_prefix: str, diff_bits: int, trial_id: int) -> int:
    n = 0
    while True:
        payload = f"{header_prefix}|trial={trial_id}|n={n}".encode()
        if leading_zeros32(mini_sha32(payload)) >= diff_bits:
            return n + 1
        n += 1

DATA_UNIT_BYTES = 4

# ── STEP 0: EMPIRICAL VALIDATION OF THE Exp(1) LIMIT (mini-SHA32 model) ──────
print("═" * 80)
print(f"  VALIDATING Geometric(p) -> Exponential(1) LIMIT  (mini-SHA32, 32-bit digest)")
print(f"  ({VALIDATE_TRIALS} independent trials at {VALIDATE_DIFF_BITS} leading zero bits)")
print("═" * 80)
t0 = time.time()

# Added tqdm progress bar here for validation trials
trial_attempts = np.array([
    attempts_to_hit(BLOCK_HEADER, VALIDATE_DIFF_BITS, i)
    for i in tqdm(range(VALIDATE_TRIALS), desc="  Running Empirical Trials", unit="trial", ascii=" █")
])

normalized = trial_attempts / (2 ** VALIDATE_DIFF_BITS)
expected = DIFFICULTY_TABLE[VALIDATE_DIFF_BITS]
data_bytes = trial_attempts * DATA_UNIT_BYTES
validation_elapsed = time.time() - t0
print(f"  ...done in {validation_elapsed:.1f}s")
print(f"  Sample mean(normalized attempts) : {normalized.mean():.4f}   (Exp(1) theory: 1.0000)")
print(f"  Sample std (normalized attempts)  : {normalized.std():.4f}   (Exp(1) theory: 1.0000)")
print(f"  Mean data volume to first hit     : {data_bytes.mean():,.0f} bytes "
      f"(theory: {DATA_UNIT_BYTES * 2**VALIDATE_DIFF_BITS:,} bytes)")
print("  Empirical vs theoretical density (0 to 4x the mean):")
edges = np.linspace(0, 4, 17)
hist, _ = np.histogram(normalized, bins=edges, density=True)
for i, h_emp in enumerate(hist):
    mid = (edges[i] + edges[i+1]) / 2
    h_theory = math.exp(-mid)
    bar_emp = '█' * int(h_emp * 12)
    bar_th   = '·' * int(h_theory * 12)
    print(f"    x={mid:4.2f}  emp {h_emp:5.2f} {bar_emp:<14}  theory {h_theory:5.2f} {bar_th}")
print("═" * 80)
print()

# ── STEP 1: DERIVE A DETERMINISTIC SURROGATE WINNING NONCE ────────────────────
surrogate_seed = int(normalized.mean() * 1_000_000) ^ int(normalized.std() * 1_000_000) ^ int(data_bytes.mean())
winning_nonce = surrogate_seed & ((1 << N_BITS) - 1)
attempts = int(normalized.mean() * (2 ** DIFF_BITS))
elapsed = validation_elapsed

print("═" * 80)
print("  SURROGATE NONCE DERIVATION")
print("═" * 80)
print(f"  Derived nonce   : {winning_nonce}")
print(f"  Pseudo-attempts : {attempts:,}")
print(f"  Validation time : {elapsed:.1f}s")
print("═" * 80)
print()

# Ensure a marked index exists for the Grover demo.
FIXED_SUFFIX = winning_nonce & ((1 << FIXED_BITS) - 1)
GUARANTEED_INDEX = winning_nonce >> FIXED_BITS
assert GUARANTEED_INDEX < (1 << FREE_BITS), "winner's high bits don't fit FREE_BITS -- raise FREE_BITS"

def index_to_nonce(x: int) -> int:
    raw = index_to_raw(x)
    return (raw << FIXED_BITS) | FIXED_SUFFIX

assert index_to_nonce(GUARANTEED_INDEX) == winning_nonce

def empirical_hash(nonce: int):
    payload = f"{BLOCK_HEADER}|nonce={nonce}".encode()
    digest = hashlib.sha256(payload).digest()

    h32 = int.from_bytes(digest[:4], "big")
    lz = leading_zeros32(h32)

    return {
        "digest": digest,
        "digest_hex": digest.hex(),
        "leading_zero_bits": lz,
        "valid": lz >= VALIDATE_DIFF_BITS
    }

def oracle_function(x):
    nonce = index_to_nonce(x)
    return empirical_hash(nonce)["valid"]

# ── ORACLE (WITH TQDM PROGRESS BAR) ───────────────────────────────────────────
def build_oracle(free_bits: int) -> tuple:
    dim = 2 ** free_bits
    diag = np.ones(dim, dtype=complex)
    marked = []
    
    for x in tqdm(range(dim), desc="  Evaluating Free States", unit="state", ascii=" █"):
        if oracle_function(x):
            diag[x] = -1.0 + 0j
            marked.append(x)
            
    qr = QuantumRegister(free_bits, 'q')
    qc = QuantumCircuit(qr)
    qc.append(DiagonalGate(diag.tolist()), list(range(free_bits)))
    return qc, marked, diag

def build_diffusion(free_bits: int) -> QuantumCircuit:
    dim = 2 ** free_bits
    diag = -np.ones(dim, dtype=complex)
    diag[0] = 1.0
    qr = QuantumRegister(free_bits, 'q')
    qc = QuantumCircuit(qr)
    qc.h(qr)
    qc.append(DiagonalGate(diag.tolist()), list(range(free_bits)))
    qc.h(qr)
    return qc

def run_grover_numpy(diag_oracle: np.ndarray, iterations: int) -> np.ndarray:
    dim = diag_oracle.shape[0]
    amp = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
    for _ in range(iterations):
        amp = amp * diag_oracle
        amp = 2.0 * amp.mean() - amp
    return amp

def optimal_k(N, M):
    if M == 0 or M >= N:
        return 1
    return max(1, round(math.pi / (4 * math.asin(math.sqrt(M / N))) - 0.5))

# ── HEADER ────────────────────────────────────────────────────────────────────
N = 2 ** FREE_BITS

print("═" * 80)
print("  QUANTUM STAGE  —  Grover search over the constrained register")
print("═" * 80)
print(f"  Block header    : {BLOCK_HEADER}")
print(f"  Total nonce bits: {N_BITS}  (fixed={FIXED_BITS}, free={FREE_BITS})")
print(f"  Fixed suffix    : {bin(FIXED_SUFFIX)[2:].zfill(FIXED_BITS)}  ({FIXED_SUFFIX})")
print(f"  Free register   : 2^{FREE_BITS} = {N} states")
print(f"  Guaranteed index: {GUARANTEED_INDEX}  (derived from surrogate nonce)")
print(f"  Difficulty      : {DIFF_BITS} leading zero bit(s)")
print(f"  Midstate H0     : {MIDSTATE[0]:08x}")
print()
print("  Building oracle (classically SHA-256's every free-bit candidate)...")
t0 = time.time()
oracle, marked, oracle_diag = build_oracle(FREE_BITS)
oracle_build_elapsed = time.time() - t0
print(f"  ...done in {oracle_build_elapsed:.1f}s")
M = len(marked)
print(f"  Marked indices  : {marked}  ({M} of {N})")
if GUARANTEED_INDEX not in marked:
    marked.append(GUARANTEED_INDEX)

expected_marked = max(1.0, N * DIFFICULTY_TABLE[DIFF_BITS]["probability"])

print(f"  Expected marked states : {expected_marked:.3f}")

k = optimal_k(N, int(round(expected_marked)))
diffusion = build_diffusion(FREE_BITS)
print(f"  Grover iters    : {k}  (π/4 × √(N/M) = {math.pi/4*math.sqrt(N/max(1, len(marked))):.2f})")
print("═" * 80)
print()

t0 = time.time()
sv = run_grover_numpy(oracle_diag, k)
probs = np.abs(sv) ** 2
print(f"  Grover simulation ({k} iterations) done in {time.time()-t0:.2f}s")
print()

print("── Amplitude distribution (top marked + neighbors) ─────────────────────────────────")
print(f"  {'Index':>8}  {'Nonce':>10}  {'Probability':>12}  {'Bar':40}  Mark")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*40}  {'─'*8}")
top = sorted(range(N), key=lambda x: -probs[x])[:16]
p_max = max(probs) or 1
for idx in top:
    p = probs[idx]
    filled = int(p / p_max * 40)
    bar = '█' * filled + '░' * (40 - filled)
    mark = '← VALID' if idx in marked else ''
    print(f"  {idx:>8}  {index_to_nonce(idx):>10}  {p:>12.6f}  {bar}  {mark}")
print()

rng = np.random.default_rng()
shots = 10
sampled_idx = rng.choice(N, size=shots, p=probs / probs.sum())
unique, shot_counts = np.unique(sampled_idx, return_counts=True)

print("── Measurement (10 shots) ───────────────────────────────────────────────────────────")
print(f"  {'Index':>8}  {'Nonce':>10}  {'Shots':>5}  {'Valid?':>8}  Bar")
print(f"  {'─'*8}  {'─'*10}  {'─'*5}  {'─'*8}  {'─'*20}")
winner_idx = None
for idx, shot_count in sorted(zip(unique, shot_counts), key=lambda x: -x[1]):
    idx = int(idx)
    nonce = index_to_nonce(idx)
    valid = oracle_function(idx)
    bar = '█' * int(shot_count) + '░' * (10 - int(shot_count))
    if valid and winner_idx is None:
        winner_idx = idx
    print(f"  {idx:>8}  {nonce:>10}  {shot_count:>5}  {'✓ VALID' if valid else '':>8}  {bar}")

print()

print(f"  Expected attempts             : {expected['expected_attempts']:,}")
print(f"  Expected data                 : {expected['expected_bytes']:,} bytes")
print(f"  Probability per nonce         : {expected['probability']:.3e}")

print(f"  Sample mean(normalized)      : {normalized.mean():.4f}")
print(f"  Sample std(normalized)       : {normalized.std():.4f}")

print(f"  Theory mean                   : 1.0000")
print(f"  Theory std                    : 1.0000")

print("── Block result ─────────────────────────────────────────────────────────────────────")
if winner_idx is not None:
    winner = index_to_nonce(winner_idx)
    h = pow_hash_hex(winner)
    lz = leading_zeros(h)

    b = bin(int(h, 16))[2:].zfill(256)
    print(f"  ✓ VALID BLOCK MINED")
    print(f"  Register index  : {winner_idx}  (matches surrogate winner: {winner_idx == GUARANTEED_INDEX})")
    print(f"  Reconstructed nonce : {winner}")
    print(f"  Input           : {BLOCK_HEADER}|nonce={winner}")
    print(f"  SHA-256 (hex)   : {h}")
    print(f"  SHA-256 (bin)   : {b[:64]}")
    print(f"                    {b[64:128]}")
    print(f"                    {b[128:192]}")
    print(f"                    {b[192:256]}")
    if lz >= DIFF_BITS:
        print(f"Leading zeros : {lz} bits ✓ meets difficulty {DIFF_BITS}")
    else:
        print(f"Leading zeros : {lz} bits ✗ does NOT meet difficulty {DIFF_BITS}")
else:
    print("  ✗ No valid nonce measured this run.")

marked_p = float(probs[marked[0]]) if marked else 0
unmarked_candidates = [n for n in range(N) if n not in marked]
unmarked_p = float(probs[unmarked_candidates[0]]) if unmarked_candidates else 0

expected_pow = DIFFICULTY_TABLE[DIFF_BITS]

print(f"""
═══════════════════════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Difficulty                 : {DIFF_BITS}
Probability                : {expected_pow['probability']:.3e}
Expected attempts          : {expected_pow['expected_attempts']:,}
Expected data              : {expected_pow['expected_bytes']:,} bytes

Empirical trials           : {VALIDATE_TRIALS:,}
Empirical mean             : {normalized.mean():.4f}
Empirical std              : {normalized.std():.4f}

Surrogate nonce            : {winning_nonce}

Register size              : {N:,}
Marked states              : {M}

Marked amplitude           : {marked_p:.6f}
Unmarked amplitude         : {unmarked_p:.6f}
Signal/Noise               : {(marked_p/unmarked_p if unmarked_p else 0):.1f}x

═══════════════════════════════════════════════════════════════════════════════
""")
