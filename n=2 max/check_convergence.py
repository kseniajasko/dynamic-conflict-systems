"""
Numerical check of global convergence for the map of maxima Phi
(Section 5.1 of "Dynamics of conflict interaction in terms of dominant
players: from oscillations to symmetry breaking").

The asymmetric attractor is
  - the exchange 2-cycle {(x*, y*), (y*, x*)} for 0 < alpha < 2,
  - the pair of winner-loser fixed points (x*, y*), (y*, x*) for alpha > 2.
At alpha = 2 it does not exist (every trajectory converges to (1/2, 1/2)),
so this value is not tested.

1. Random initial points.
   For each alpha in ALPHAS, N = 10^5 initial points (x0, y0) uniformly
   distributed in (1/2, 1)^2 are iterated by Phi, up to 6000 steps.
   A trajectory is counted as convergent at the first even step at which
   its distance to the asymmetric attractor is less than 1e-10.
   Reported: fraction of convergent trajectories, median and maximal number
   of steps; for alpha > 2 also the number of trajectories whose limit is
   not (x*, y*) when x0 > y0 (resp. (y*, x*) when x0 < y0), which must be 0.

2. Regular grid.
   For alpha = 0.5, 1.5, 2.5 the 400 x 399 grid of the basin figure
   (159 600 points, none on the diagonal) is iterated 4000 steps, and the
   points that are not within 1e-6 of a point of the asymmetric attractor
   are counted (must be 0).

The results are printed and written to out/convergence_check.txt.

Run:  python3 check_convergence.py              (seed 2026, as in the paper)
      python3 check_convergence.py --seed 1     (another random sample)
      python3 check_convergence.py --n 10000    (smaller sample)
Requires: numpy.
"""
import argparse
import os
import time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ALPHAS = [0.2, 0.5, 1.0, 1.5, 1.9, 1.99, 2.01, 2.1, 2.5, 3.0, 5.0, 10.0]
MAX_STEPS = 6000
TOL = 1e-10
GRID_ALPHAS = [0.5, 1.5, 2.5]
GRID_STEPS = 4000
GRID_TOL = 1e-6


# ---------------------------------------------------------------- model
def Phi(x, y, a):
    """Map of maxima (eq_macro_system); vectorised."""
    pa, pp = a * x * (1 - y), 1 - x
    ra, rp = a * y * (1 - x), 1 - y
    return np.maximum(pa, pp) / (pa + pp), np.maximum(ra, rp) / (ra + rp)


def cycle4(a):
    """Exchange 2-cycle (x*, y*) for 0 < alpha < 2: root in (1/2, 1) of
    alpha^3 x^3 + alpha x^2 + (1 - alpha) x - 1 = 0, y* = 1 / (1 + alpha x*)."""
    roots = np.roots([a**3, a, 1 - a, -1])
    x = [t.real for t in roots if abs(t.imag) < 1e-12 and 0.5 < t.real < 1][0]
    return x, 1 / (1 + a * x)


def fixed_asym(a):
    """Winner-loser fixed point (x*, y*) for alpha > 2."""
    return 1 - 1 / (a * (a - 1) ** 2), 1 - 1 / a


def attractor(a):
    return cycle4(a) if a < 2 else fixed_asym(a)


# ---------------------------------------------------------------- 1. random initial points
def random_check(a, rng, N):
    x = rng.uniform(0.5, 1, N); y = rng.uniform(0.5, 1, N)
    x0, y0 = x.copy(), y.copy()
    u, v = attractor(a)
    hit = np.full(N, -1)
    for t in range(1, MAX_STEPS + 1):
        x, y = Phi(x, y, a)
        if t % 2 == 0:
            d = np.minimum(np.hypot(x - u, y - v), np.hypot(x - v, y - u))
            hit[(hit < 0) & (d < TOL)] = t
            if (hit > 0).all():          # all trajectories have converged
                break
    conv = hit > 0
    wrong = None
    if a > 2:
        at_winner_first = np.hypot(x - u, y - v) < 1e-8      # limit is (x*, y*)
        wrong = int(np.sum(conv & (at_winner_first != (x0 > y0))))
    med = int(np.median(hit[conv])) if conv.any() else None
    mx = int(hit[conv].max()) if conv.any() else None
    return conv.mean(), med, mx, wrong


# ---------------------------------------------------------------- 2. regular grid
def grid_check(a, n=400):
    gx = np.linspace(0.5, 1, n + 2)[1:-1]
    gy = np.linspace(0.5, 1, n + 1)[1:]
    gy = gy[gy < 1]                       # offset grid: avoids the diagonal
    x, y = np.meshgrid(gx, gy)
    for _ in range(GRID_STEPS):
        x, y = Phi(x, y, a)
    u, v = attractor(a)
    ok = ((abs(x - u) < GRID_TOL) & (abs(y - v) < GRID_TOL)) | \
         ((abs(x - v) < GRID_TOL) & (abs(y - u) < GRID_TOL))
    return x.size, int((~ok).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=2026,
                    help="seed of numpy default_rng (default 2026, as in the paper)")
    ap.add_argument("--n", type=int, default=100000,
                    help="number of random initial points per alpha (default 100000)")
    args = ap.parse_args()

    t0 = time.time()
    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("1. Random initial points: N = %d per alpha, seed = %d, up to %d steps,"
        % (args.n, args.seed, MAX_STEPS))
    out("   tolerance %g at even steps" % TOL)
    out()
    out("   alpha   converged   median steps   max steps   wrong winner (alpha > 2)")
    rng = np.random.default_rng(args.seed)
    all_ok = True
    for a in ALPHAS:
        fr, med, mx, wrong = random_check(a, rng, args.n)
        all_ok &= (fr == 1.0) and (wrong in (None, 0))
        out("   %5g   %8.3f %%   %12s   %9s   %s"
            % (a, 100 * fr, med, mx, "--" if wrong is None else wrong))
    out()
    out("2. Regular 400 x 399 grid, %d steps, tolerance %g" % (GRID_STEPS, GRID_TOL))
    out()
    for a in GRID_ALPHAS:
        size, bad = grid_check(a)
        all_ok &= (bad == 0)
        out("   alpha = %g: grid points = %d, not converged to the asymmetric attractor = %d"
            % (a, size, bad))
    out()
    out("All checks passed." if all_ok else "SOME CHECKS FAILED.")
    with open(os.path.join(OUT, "convergence_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written to %s (%.0f s)" % (os.path.join(OUT, "convergence_check.txt"),
                                      time.time() - t0))


if __name__ == "__main__":
    main()