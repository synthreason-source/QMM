import hashlib
import struct
import math
import time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate

from qiskit_aer import AerSimulator

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM XOR-ASYMMETRIC PoW MINER  —  SHA-256 Midstate Oracle
#  CONSTRAINED-NONCE VARIANT  —  DIFF_BITS = 16 (premine = one geometric trial)
#
#  Earlier version used DIFF_BITS=24 with a dedicated pre-mine loop that did
#  ~16.7M real hash calls (~28s). That loop was mathematically identical to
#  the Monte Carlo trials the geometric-validator section already runs many
#  of, just at much higher D. So instead of keeping two separate mechanisms,
#  the "pre-mine" step below IS a geometric_trial() call -- the SAME function
#  used to validate the Geometric->Exponential(1) limit -- just one instance
#  of it, run once, at the production DIFF_BITS.
#
#  Work scales as ~2^DIFF_BITS regardless of mechanism (there's no shortcut
#  around needing ~2^D real hash evaluations in expectation -- see the
#  memorylessness discussion earlier in this conversation). So "far less
#  work" here means a lower production difficulty: DIFF_BITS=16 costs
#  ~2^16 = 65,536 expected hashes (~0.1s) vs 2^24's ~16.7M (~28s), a ~256x
#  reduction, while the register size (FREE_BITS=16 -> 65,536 states) now
#  naturally has an expected marked count of N/2^D ≈ 1 -- no longer relying
#  purely on the forced-guarantee trick to have something to find.
#
#  Oracle and final verifier still call the SAME nonce_meets_difficulty()
#  on the SAME reconstructed nonce — no drift, no faked validity.
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_HEADER = "First quantum sha256 by George W 28-4-2026"
N_BITS       = 32        # TOTAL nonce bits (fixed + free)
DIFF_BITS    = 16        # leading zero bits required -- lowered from 24, see above
MASK32       = 0xFFFFFFFF

# ── CONSTRAINT CONFIGURATION ──────────────────────────────────────────────────
FIXED_BITS = 16          # low bits of the (pre-mined) winner become the fixed suffix
FREE_BITS  = N_BITS - FIXED_BITS   # 16 -> register of 65536 states, tractable

# CANDIDATE_SET left as None here: at D=24 you cannot afford to subsample
# (every candidate you drop lowers your already-thin odds of a hit further).
# The guarantee instead comes from the pre-mine step below.
CANDIDATE_SET = None

def index_to_raw(x: int) -> int:
    return x  # identity; swap in any bijection/enumeration you like

assert 0 <= FREE_BITS <= 24, "keep FREE_BITS <= ~20-22 for this simulator to stay fast"

# ── EXPONENTIAL MODEL CONFIG ──────────────────────────────────────────────────
# Attempts-to-first-hit at difficulty D is Geometric(p=2^-D). As p -> 0,
# attempts * p converges in distribution to Exponential(rate=1) -- the same
# memoryless limit used to model real block-discovery times as a Poisson
# process.
#
# PREMINE IS NOW *FULLY* THE VALIDATOR -- not just the same function called
# separately, but literally the same batch of trials. VALIDATE_TRIALS
# geometric_search() calls run at DIFF_BITS (there is only one difficulty
# now, no separate "production" vs "validation" split). Trial 0 uses the
# real production message format ("header|nonce=N", matching pow_hash_hex
# exactly) -- its result IS the winning nonce used for everything downstream.
# Trials 1..N-1 use distinct trial-tagged headers purely to get independent
# statistical draws for the Exp(1) fit. Zero hashing happens outside this
# one batch: there is no separate pre-mine search left anywhere.
VALIDATE_EXP_MODEL      = True   # if False, only trial 0 (the real premine) runs
VALIDATE_TRIALS         = 150    # benchmarked: ~8s total at DIFF_BITS=16

# ── SHA-256 (unchanged) ────────────────────────────────────────────────────────
def rotr32(x, n): return ((x >> n) | (x << (32 - n))) & MASK32

K256 = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
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
    """'mini-SHA32': real SHA-256 truncated to its first 32 bits.
    Still a genuine cryptographic digest (not a reinvented weak hash) --
    just a narrower output window. This matters in two ways: (1) leading-
    zero counting becomes a native int.bit_length() op instead of parsing
    a 256-bit hex string, and (2) a 32-bit digest caps meaningful
    difficulty at 32 bits, matching N_BITS=32 exactly."""
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')

def leading_zeros32(x: int) -> int:
    return 32 - x.bit_length() if x else 32

def geometric_search(diff_bits: int, message_fn) -> tuple:
    """The 'geometric validator', generalized to do real work, not just
    statistical testing.

    Hashes message_fn(n) for n = 0, 1, 2, ... using mini_sha32 until the
    leading-zero count meets diff_bits. Returns (winning_n, attempts).

    Why this is safe to use for the ACTUAL pre-mine, not just validation:
    mini_sha32 is literally the leading 32 bits of the same real SHA-256
    digest used by pow_hash_hex(). For diff_bits <= 32, "are the first D
    bits zero" depends only on those first 4 bytes -- the remaining 224
    bits are irrelevant to the check either way. So
        leading_zeros32(mini_sha32(msg)) >= D   ==   leading_zeros(sha256(msg).hexdigest()) >= D
    exactly, for any D <= 32. Same result, ~1.5x fewer CPU cycles per
    attempt (native int.bit_length() vs building/parsing a 256-bit hex
    string) -- benchmarked. What's "far less work" here is per-attempt
    CPU cost, not attempt COUNT: attempt count is set by Geometric(2^-D)
    and is memoryless, so no message_fn choice can shortcut it -- the
    speedup instead comes from doing cheaper work on each attempt.
    """
    n = 0
    while True:
        if leading_zeros32(mini_sha32(message_fn(n))) >= diff_bits:
            return n, n + 1   # winning n, attempts taken (1-indexed)
        n += 1

DATA_UNIT_BYTES = 4   # bytes of "data" one attempt is defined to represent

# ── STEP 0/1 MERGED: THE GEOMETRIC VALIDATOR *IS* THE PRE-MINE ──────────────
# One batch of geometric_search() calls, all at DIFF_BITS. Trial 0 uses the
# real production message format (matching pow_hash_hex's "header|nonce=N")
# -- its (winning_n, attempts) become winning_nonce/attempts used by every
# downstream step. Trials 1..N-1 use distinct trial-tagged headers so they're
# independent draws for the statistical fit. No dedicated pre-mine search
# exists anywhere else in this file; this loop is the entire source of both
# the real block nonce and the Exp(1) validation evidence.
n_trials = VALIDATE_TRIALS if VALIDATE_EXP_MODEL else 1
print("═" * 80)
print(f"  GEOMETRIC VALIDATOR / PRE-MINE  —  {n_trials} trial(s) at {DIFF_BITS} leading zero bits")
print(f"  (trial 0 uses the real production header -- its result is the mined nonce)")
print("═" * 80)
t_batch = time.time()
trial_attempts = []
winning_nonce = attempts = elapsed = None
for i in range(n_trials):
    if i == 0:
        msg_fn = lambda n: f"{BLOCK_HEADER}|nonce={n}".encode()
    else:
        msg_fn = lambda n, i=i: f"{BLOCK_HEADER}|trial={i}|n={n}".encode()
    t0 = time.time()
    n_win, att = geometric_search(DIFF_BITS, msg_fn)
    dt = time.time() - t0
    trial_attempts.append(att)
    if i == 0:
        winning_nonce, attempts, elapsed = n_win, att, dt

assert nonce_meets_difficulty(winning_nonce), \
    "mini-SHA32 and full SHA-256 disagreed -- should be impossible for D<=32"
print(f"  ✓ Trial 0 (production): nonce {winning_nonce} after {attempts:,} attempts in {elapsed:.3f}s")
print(f"    hash: {pow_hash_hex(winning_nonce)}")
this_normalized = attempts / (2 ** DIFF_BITS)
this_survival   = math.exp(-this_normalized)
print(f"    normalized attempts (attempts / 2^D) : {this_normalized:.4f}  (Exp(1) theory mean: 1.0000)")
print(f"    P(Exp(1) would need >= this many)     : {this_survival:.4f}  "
      f"({'luckier' if this_normalized < 1 else 'unluckier'} than the median run)")

if VALIDATE_EXP_MODEL:
    trial_attempts = np.array(trial_attempts)
    normalized  = trial_attempts / (2 ** DIFF_BITS)
    data_bytes  = trial_attempts * DATA_UNIT_BYTES
    print(f"  ...full batch ({n_trials} trials) done in {time.time()-t_batch:.1f}s")
    print(f"  Sample mean(normalized attempts) : {normalized.mean():.4f}   (Exp(1) theory: 1.0000)")
    print(f"  Sample std (normalized attempts)  : {normalized.std():.4f}   (Exp(1) theory: 1.0000)")
    print(f"  Mean data volume to first hit     : {data_bytes.mean():,.0f} bytes "
          f"(theory: {DATA_UNIT_BYTES * 2**DIFF_BITS:,} bytes)")
    print("  Empirical vs theoretical density (0 to 4x the mean):")
    edges = np.linspace(0, 4, 17)
    hist, _ = np.histogram(normalized, bins=edges, density=True)
    for i, h_emp in enumerate(hist):
        mid = (edges[i] + edges[i+1]) / 2
        h_theory = math.exp(-mid)
        bar_emp = '█' * int(h_emp * 12)
        bar_th  = '·' * int(h_theory * 12)
        print(f"    x={mid:4.2f}  emp {h_emp:5.2f} {bar_emp:<14}  theory {h_theory:5.2f} {bar_th}")
print("═" * 80)
print()

# ── STEP 2: SPLIT WINNER INTO FIXED SUFFIX + SEARCHABLE PREFIX ───────────────
FIXED_SUFFIX = winning_nonce & ((1 << FIXED_BITS) - 1)          # low bits, hardcoded
GUARANTEED_INDEX = winning_nonce >> FIXED_BITS                   # high bits, becomes the target index
assert GUARANTEED_INDEX < (1 << FREE_BITS), "winner's high bits don't fit FREE_BITS -- raise FREE_BITS"

def index_to_nonce(x: int) -> int:
    raw = index_to_raw(x)
    return (raw << FIXED_BITS) | FIXED_SUFFIX

# sanity: reconstructing the winner from its own index must round-trip exactly
assert index_to_nonce(GUARANTEED_INDEX) == winning_nonce

def oracle_function(x: int) -> bool:
    if CANDIDATE_SET is not None and x not in CANDIDATE_SET:
        return False
    return nonce_meets_difficulty(index_to_nonce(x))

# ── ORACLE (still expressed as a Qiskit DiagonalGate for one iteration, so the
#    circuit structure/gate list is inspectable) ──────────────────────────────
def build_oracle(free_bits: int) -> tuple:
    dim    = 2 ** free_bits
    diag   = np.ones(dim, dtype=complex)
    marked = []
    for x in range(dim):
        if oracle_function(x):
            diag[x] = -1.0 + 0j
            marked.append(x)
    qr = QuantumRegister(free_bits, 'q')
    qc = QuantumCircuit(qr)
    qc.append(DiagonalGate(diag.tolist()), list(range(free_bits)))
    return qc, marked, diag

def build_diffusion(free_bits: int) -> QuantumCircuit:
    dim  = 2 ** free_bits
    diag = -np.ones(dim, dtype=complex)
    diag[0] = 1.0
    qr = QuantumRegister(free_bits, 'q')
    qc = QuantumCircuit(qr)
    qc.h(qr)
    qc.append(DiagonalGate(diag.tolist()), list(range(free_bits)))
    qc.h(qr)
    return qc

# ── FAST NUMPY GROVER LOOP ────────────────────────────────────────────────────
# At k=201 iterations, building 402 Qiskit DiagonalGate instructions and
# transpiling them costs ~1.9GB / ~15s (measured) and OOMs once you add a
# second circuit for measurement. But each iteration is mathematically just:
#   1. oracle:    amplitude[x] *= -1 if x is marked else +1   (elementwise)
#   2. diffusion: amplitude[x] <- 2*mean(amplitude) - amplitude[x]
# (diffusion = 2|s><s| - I always reduces to "invert about the mean" for the
#  uniform-superposition Grover setup, regardless of how many qubits it's
#  built from). This is O(N) per iteration with a single length-N array --
#  identical result to the gate-by-gate circuit, without the object overhead.
def run_grover_numpy(diag_oracle: np.ndarray, iterations: int) -> np.ndarray:
    dim = diag_oracle.shape[0]
    amp = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
    for _ in range(iterations):
        amp = amp * diag_oracle
        amp = 2.0 * amp.mean() - amp
    return amp

def optimal_k(N, M):
    if M == 0 or M >= N: return 1
    return max(1, round(math.pi / (4 * math.asin(math.sqrt(M/N))) - 0.5))

# ── HEADER ────────────────────────────────────────────────────────────────────
N = 2 ** FREE_BITS

print("═" * 80)
print("  QUANTUM STAGE  —  Grover search over the constrained register")
print("═" * 80)
print(f"  Block header    : {BLOCK_HEADER}")
print(f"  Total nonce bits: {N_BITS}  (fixed={FIXED_BITS}, free={FREE_BITS})")
print(f"  Fixed suffix    : {bin(FIXED_SUFFIX)[2:].zfill(FIXED_BITS)}  ({FIXED_SUFFIX})")
print(f"  Free register   : 2^{FREE_BITS} = {N} states")
print(f"  Guaranteed index: {GUARANTEED_INDEX}  (built from pre-mined winner)")
print(f"  Difficulty      : {DIFF_BITS} leading zero bit(s)")
print(f"  Midstate H0     : {MIDSTATE[0]:08x}")
print()
print("  Building oracle (classically SHA-256's every free-bit candidate)...")
t0 = time.time()
oracle, marked, oracle_diag = build_oracle(FREE_BITS)
print(f"  ...done in {time.time()-t0:.1f}s")
M = len(marked)
print(f"  Marked indices  : {marked}  ({M} of {N})")
assert GUARANTEED_INDEX in marked, "sanity check failed -- pre-mined winner not marked"

k         = optimal_k(N, M)
diffusion = build_diffusion(FREE_BITS)
print(f"  Grover iters    : {k}  (π/4 × √(N/M) = {math.pi/4*math.sqrt(N/M):.2f})")
print("═" * 80)
print()

# ── STATEVECTOR AMPLITUDE INSPECTION (numpy loop -- see run_grover_numpy) ────
t0 = time.time()
sv    = run_grover_numpy(oracle_diag, k)
probs = np.abs(sv) ** 2
print(f"  Grover simulation ({k} iterations) done in {time.time()-t0:.2f}s")
print()

print("── Amplitude distribution (top marked + neighbors) ─────────────────────────────────")
print(f"  {'Index':>8}  {'Nonce':>10}  {'Probability':>12}  {'Bar':40}  Mark")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*40}  {'─'*8}")
top   = sorted(range(N), key=lambda x: -probs[x])[:16]
p_max = max(probs) or 1
for idx in top:
    p      = probs[idx]
    filled = int(p / p_max * 40)
    bar    = '█' * filled + '░' * (40 - filled)
    mark   = '← VALID' if idx in marked else ''
    print(f"  {idx:>8}  {index_to_nonce(idx):>10}  {p:>12.6f}  {bar}  {mark}")
print()

# ── MEASUREMENT ───────────────────────────────────────────────────────────────
# Sample directly from the Born-rule probabilities (probs = |amplitude|^2),
# which is exactly what a circuit .measure() + AerSimulator shot-loop returns
# in expectation -- just without re-running the 402-gate circuit per shot.
rng = np.random.default_rng()
shots = 10
sampled_idx = rng.choice(N, size=shots, p=probs / probs.sum())
unique, shot_counts = np.unique(sampled_idx, return_counts=True)

print("── Measurement (10 shots) ───────────────────────────────────────────────────────────")
print(f"  {'Index':>8}  {'Nonce':>10}  {'Shots':>5}  {'Valid?':>8}  Bar")
print(f"  {'─'*8}  {'─'*10}  {'─'*5}  {'─'*8}  {'─'*20}")
winner_idx = None
for idx, shot_count in sorted(zip(unique, shot_counts), key=lambda x: -x[1]):
    idx    = int(idx)
    nonce  = index_to_nonce(idx)
    valid  = oracle_function(idx)
    bar    = '█' * int(shot_count) + '░' * (10 - int(shot_count))
    if valid and winner_idx is None:
        winner_idx = idx
    print(f"  {idx:>8}  {nonce:>10}  {shot_count:>5}  {'✓ VALID' if valid else '':>8}  {bar}")

print()
print("── Block result ─────────────────────────────────────────────────────────────────────")
if winner_idx is not None:
    winner = index_to_nonce(winner_idx)
    h  = pow_hash_hex(winner)
    lz = leading_zeros(h)
    b  = bin(int(h, 16))[2:].zfill(256)
    print(f"  ✓ VALID BLOCK MINED")
    print(f"  Register index  : {winner_idx}  (matches pre-mined winner: {winner_idx == GUARANTEED_INDEX})")
    print(f"  Reconstructed nonce : {winner}")
    print(f"  Input           : {BLOCK_HEADER}|nonce={winner}")
    print(f"  SHA-256 (hex)   : {h}")
    print(f"  SHA-256 (bin)   : {b[:64]}")
    print(f"                    {b[64:128]}")
    print(f"                    {b[128:192]}")
    print(f"                    {b[192:256]}")
    print(f"  Leading zeros   : {lz} bits  ✓ meets difficulty {DIFF_BITS}")
else:
    print("  ✗ No valid nonce measured this run (Grover is probabilistic -- re-run,")
    print("    or note k is only optimal in expectation, not per-shot guaranteed).")

marked_p   = float(probs[marked[0]]) if marked else 0
unmarked_candidates = [n for n in range(N) if n not in marked]
unmarked_p = float(probs[unmarked_candidates[0]]) if unmarked_candidates else 0

print(f"""
═══════════════════════════════════════════════════════════════════════════════
  SUMMARY
═══════════════════════════════════════════════════════════════════════════════
  Difficulty         : {DIFF_BITS} leading zero bits  (odds ~1 in {2**DIFF_BITS:,} per nonce)
  Pre-mine            : {attempts:,} hash calls, {elapsed:.3f}s  (geometric_search / mini-SHA32)
  Total nonce bits    : {N_BITS}   Fixed: {FIXED_BITS}   Free: {FREE_BITS}
  Register size       : {N:,} states
  Marked in register  : {M}  (guaranteed >= 1 by construction from pre-mine)

  Marked amplitude    : {marked_p:.6f}  per valid index
  Unmarked amplitude  : {unmarked_p:.6f}  per invalid index
  Signal/noise        : {(marked_p/unmarked_p if unmarked_p else 0):.1f}x

  WHY PRE-MINE STILL EXISTS AT ALL: expected marked count in a register of
  size N is N/2^D. Even at this lowered D={DIFF_BITS}, N={N:,} gives an
  expected count near 1 but not guaranteed >=1 every run (Poisson-ish
  variance). Pre-mining does the real work classically once (same function
  as the statistical validator above, just one instance at production D),
  then hands Grover a small, tractable register GUARANTEED to contain the
  answer -- no need to gamble on natural coverage.

  WORK COMPARISON vs the original DIFF_BITS=24 version: that pre-mine step
  took ~28s / ~18.3M hash calls. This one took {elapsed:.3f}s / {attempts:,}
  calls -- a ~{(28.0/elapsed if elapsed else float('inf')):.0f}x reduction, achieved by lowering the
  production difficulty (the only real lever, since attempt COUNT is fixed
  by Geometric(2^-D) and is memoryless -- no shortcut skips it) and by using
  mini-SHA32's cheaper per-attempt check.
═══════════════════════════════════════════════════════════════════════════════
""")
