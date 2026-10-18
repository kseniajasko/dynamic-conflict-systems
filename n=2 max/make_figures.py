"""
Figures for "Dynamics of conflict interaction in terms of dominant players".

Colour code (meaning-carrying colours):
  teal        = oscillatory regime (both maxima visit both values in time)
  blue        = vector p / winner branch
  red         = vector r / loser branch
  grey dashed = symmetric state (saddle)
  purple      = multipliers / contraction rates
All figures are written as vector PDF (plus PNG previews) into ./out.
The only random initial points (collapse figure) use numpy default_rng(7);
everything else is deterministic.
The numerical checks of global convergence (Section 5.1) are performed by
the separate script check_convergence.py.

Run:  python3 make_figures.py
Requires: numpy, matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
TEAL = "#117a65"; BLUE = "#1f618d"; RED = "#c0392b"; GREY = "#909497"
PURPLE = "#8e44ad"
SPAN_L = "#eaf2f8"; SPAN_R = "#fdf2e9"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
W = 6.5   # full text width, inches


def save(fig, name):
    axes = [ax for ax in fig.axes if ax.get_title()]
    if len(axes) > 1:                      # panel letters, referred to in the captions
        for k, ax in enumerate(axes):
            ax.set_title("(%s) " % "abcdef"[k] + ax.get_title())
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def grid(ax, dotted=False):
    if dotted:
        ax.grid(True, ls=":", alpha=0.5)
    else:
        ax.grid(alpha=0.15)


def spans(ax, a_c, a_max):
    ax.axvspan(0, a_c, color=SPAN_L, alpha=0.55, lw=0)
    ax.axvspan(a_c, a_max, color=SPAN_R, alpha=0.55, lw=0)
    ax.axvline(a_c, color="k", lw=1.0)


# ---------------------------------------------------------------- model
def Phi(x, y, a):
    """Map of maxima (eq_macro_system); vectorised."""
    pa, pp = a * x * (1 - y), 1 - x
    ra, rp = a * y * (1 - x), 1 - y
    return np.maximum(pa, pp) / (pa + pp), np.maximum(ra, rp) / (ra + rp)


def full_step(p, r, a):
    """One step of the full system (conf_max) for any n; renormalised,
    since rounding errors transverse to the simplex are amplified."""
    i, j = int(np.argmax(p)), int(np.argmax(r))
    pm, rm = p[i], r[j]
    zp = a * pm * (1 - rm) + (1 - pm)
    zr = a * rm * (1 - pm) + (1 - rm)
    pn, rn = p / zp, r / zr
    pn[i] = a * pm * (1 - rm) / zp
    rn[j] = a * rm * (1 - pm) / zr
    return pn / pn.sum(), rn / rn.sum()


def cycle4(a):
    roots = np.roots([a**3, a, 1 - a, -1])
    x = [t.real for t in roots if abs(t.imag) < 1e-12 and 0.5 < t.real < 1][0]
    return x, 1 / (1 + a * x)


def xsym(a):
    return (-1 + np.sqrt(1 + 4 * a)) / (2 * a) if a < 2 else 1 - 1 / a


def fixed_asym(a):
    return 1 - 1 / (a * (a - 1) ** 2), 1 - 1 / a


def rho2(a):
    return (a * a - a + 1) / (a * a * (a - 1))


A_LO = np.linspace(0.04, 1.999, 500)
A_HI = np.linspace(2.0005, 5.0, 500)
X4 = np.array([cycle4(a) for a in A_LO])
XF = np.array([fixed_asym(a) for a in A_HI])


# ================================================================ phase space
def regime_labels(ax, a):
    g = np.linspace(0.505, 0.995, 300)
    X, Y = np.meshgrid(g, g)
    act_p = a * X * (1 - Y) > 1 - X
    act_r = a * Y * (1 - X) > 1 - Y
    for mp, mr, txt in ((1, 1, "A/A"), (1, 0, "A/P"), (0, 1, "P/A"), (0, 0, "P/P")):
        m = (act_p == mp) & (act_r == mr)
        if m.sum() < 400:
            continue
        e = m.copy()
        while True:
            e2 = e.copy()
            e2[1:, :] &= e[:-1, :]; e2[:-1, :] &= e[1:, :]
            e2[:, 1:] &= e[:, :-1]; e2[:, :-1] &= e[:, 1:]
            e2[0, :] = e2[-1, :] = e2[:, 0] = e2[:, -1] = False
            if e2.sum() < 5:
                break
            e = e2
        xs, ys = X[e], Y[e]
        k = len(xs) // 2
        if txt == "A/A" and a < 2:          # thin sliver: label with a pointer
            ax.annotate(txt, xy=(0.86, 0.865), xytext=(0.74, 0.95), fontsize=8.5,
                        color="#555", ha="center", va="center", weight="bold",
                        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"))
            continue
        if txt == "A/A":                    # keep the label off the diagonal
            xs_, ys_ = 0.80, 0.735
        else:
            xs_, ys_ = xs[k], ys[k]
        ax.text(xs_, ys_, txt, fontsize=8.5, color="#555", ha="center",
                va="center", weight="bold")


fig, axs = plt.subplots(1, 2, figsize=(W, 3.55))
for ax, a in ((axs[0], 1.5), (axs[1], 2.5)):
    x = np.linspace(0.5, 1, 600)
    yTp = 1 - (1 - x) / (a * x)
    ax.plot(x, yTp, "-", c=BLUE, lw=2.0, label=r"$T_p:\ \alpha x(1-y)=1-x$")
    ax.plot(yTp, x, "--", c=RED, lw=2.0, label=r"$T_r:\ \alpha y(1-x)=1-y$")
    if a < 2:
        ax.plot(x, 1 / (a * a * x), "-.", c=TEAL, lw=2.0, label=r"$S:\ \alpha^2xy=1$")
    ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.9)
    if a < 2:                               # T_p, T_r and S meet at (1/a, 1/a)
        ax.plot(1 / a, 1 / a, "o", ms=5, c="k", zorder=5)
    regime_labels(ax, a)
    ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_title(r"Switching curves, $\alpha=%.1f$" % a)
    grid(ax, dotted=True)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(w_pad=1.5, rect=[0, 0.07, 1, 1])
save(fig, "fig_phase_space")

# ================================================================ bifurcation diagram
fig, ax = plt.subplots(figsize=(W, 3.9))
spans(ax, 2, 5)
ax.axhline(0.5, color="gray", lw=0.7, ls=":")
ax.plot(A_LO, X4[:, 0], "-", c=TEAL, lw=2.4)
ax.plot(A_LO, X4[:, 1], "-", c=TEAL, lw=2.4)
ax.plot(A_LO, [xsym(a) for a in A_LO], "--", c=GREY, lw=1.5)
ax.plot(A_HI, XF[:, 0], "-", c=BLUE, lw=2.6)
ax.plot(A_HI, XF[:, 1], "-", c=RED, lw=2.6)
ax.plot(A_HI, 1 - 1 / A_HI, ":", c=GREY, lw=1.3)
ax.plot(2, 0.5, "o", ms=9, mfc="white", mec="k", mew=1.6, zorder=6)
aa = 0.6; xa, ya = cycle4(aa)
ax.annotate("", xy=(aa, xa), xytext=(aa, ya),
            arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.5))
ax.text(aa - 0.06, (xa + ya) / 2, "oscillation\namplitude", fontsize=7.5,
        ha="right", va="center", color="#0e6655")
ab = 3.2; xb, yb = fixed_asym(ab)
ax.annotate("", xy=(ab, xb), xytext=(ab, yb),
            arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.5))
ax.text(ab + 0.07, (xb + yb) / 2, "symmetry\ngap", fontsize=7.5, ha="left",
        va="center", color="#6c3483")
ax.text(1.0, 1.022, r"OSCILLATION  $(0<\alpha<2)$", fontsize=9, ha="center",
        weight="bold", color="#1a5276")
ax.text(3.5, 1.022, r"SYMMETRY BREAKING  $(\alpha>2)$", fontsize=9, ha="center",
        weight="bold", color="#9c640c")
ax.text(1.32, 0.71, "4-cycle: the maxima\nexchange two values", fontsize=7.5,
        ha="center", color="#0e6655")
ax.text(3.7, 0.885, r"winner  $x^*=1-\frac{1}{\alpha(\alpha-1)^2}$", fontsize=8,
        ha="center", color=BLUE)
ax.text(4.0, 0.60, r"loser  $y^*=1-\frac{1}{\alpha}$" + "\n(= saddle value)", fontsize=8,
        ha="center", color=RED)
ax.annotate("symmetric state (saddle)", xy=(0.8, xsym(0.8)), xytext=(0.12, 0.525),
            fontsize=7, ha="left", color="#707b7c",
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax.annotate(r"collapse: $x^t, y^t\to\frac{1}{2}$", xy=(2, 0.5), xytext=(2.2, 0.54),
            fontsize=8, weight="bold", arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
ax.set_xlabel(r"external-resource parameter $\alpha$")
ax.set_ylabel(r"asymptotic maxima $x^t=p^t_{\max},\ y^t=r^t_{\max}$")
ax.set_title(r"Bifurcation diagram: the two branches merge at $\alpha=2$ and split apart")
ax.set_xlim(0, 5); ax.set_ylim(0.47, 1.05)
save(fig, "fig_bif_main")

# ================================================================ mechanism
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
a1.plot(A_LO, [xsym(a) for a in A_LO], "-", c=TEAL, lw=2.0, label="switching branch (period 2)")
a1.plot(A_HI, [xsym(a) for a in A_HI], "-", c=BLUE, lw=2.0, label="preserving branch (period 1)")
a1.axvline(2, color="k", lw=0.8); a1.axhline(0.5, color="gray", lw=0.6, ls=":")
a1.plot(2, 0.5, "ko", ms=5)
a1.annotate("border collision\n" + r"(kink at $\alpha=2$)", xy=(2, 0.5), xytext=(2.45, 0.6),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
a1.text(0.75, 0.78, r"$\alpha x<1$", fontsize=8, color=TEAL)
a1.text(3.5, 0.68, r"$\alpha x>1$", fontsize=8, color=BLUE)
a1.set_xlabel(r"$\alpha$"); a1.set_ylabel(r"$x_{\mathrm{sym}}(\alpha)$")
a1.set_title(r"Symmetric state: V-shape, minimum at $\alpha=2$")
a1.set_xlim(0, 5); a1.set_ylim(0.47, 1.0)
a1.legend(loc="upper center", framealpha=0.95); grid(a1)

spans(a2, 2, 5)
a2.plot(A_LO, 1 - X4[:, 0] * X4[:, 1], "-", c=TEAL, lw=2.0, label=r"$1-x^*y^*$ (4-cycle)")
a2.plot(A_HI, rho2(A_HI), "-", c=PURPLE, lw=2.0, label=r"$\rho^2(\alpha)$ (winner-loser point)")
a2.axhline(0.75, color="k", lw=0.6, ls=":")
a2.plot(2, 0.75, "o", ms=6, mfc="white", mec="k", mew=1.2, zorder=5)
a2.annotate(r"$3/4$ from both sides", xy=(2, 0.75), xytext=(2.5, 0.83), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
a2.set_xlabel(r"$\alpha$"); a2.set_ylabel("contraction over two steps")
a2.set_title("Two-step contraction of the attractor")
a2.set_xlim(0, 5); a2.set_ylim(0, 0.95)
a2.legend(loc="lower right", framealpha=0.95)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_bif_mech")


# ================================================================ coordinate-wise diagrams
def coord_diagram(p0, r0, alphas, n_trans=3000, n_keep=8):
    P, R = [], []
    for a in alphas:
        p, r = np.array(p0, float), np.array(r0, float)
        for _ in range(n_trans):
            p, r = full_step(p, r, a)
        pk, rk = [], []
        for _ in range(n_keep):
            p, r = full_step(p, r, a)
            pk.append(p[0]); rk.append(r[0])
        P.append(pk); R.append(rk)
    return np.array(P), np.array(R)


AL = np.linspace(0.02, 5.0, 600)


def coord_axes(ax, ylab, title):
    ax.axvline(2.0, color="k", ls="--", lw=1.0, label=r"$\alpha=2$ (bifurcation)")
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_xlim(0, 5); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(ylab); ax.set_title(title)
    grid(ax, dotted=True)


P, R = coord_diagram([0.81, 0.19], [0.11, 0.89], AL)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.7))
a1.scatter(np.repeat(AL, P.shape[1]), P.ravel(), s=0.8, c=BLUE, marker=",", lw=0)
a2.scatter(np.repeat(AL, R.shape[1]), R.ravel(), s=0.8, c=RED, marker=",", lw=0)
coord_axes(a1, r"$p_1$", r"coordinate $p_1$")
coord_axes(a2, r"$r_1$", r"coordinate $r_1$")
a1.legend(loc="upper right", framealpha=0.95)
fig.suptitle(r"Asymmetric start $\mathbf{p}^0=(0.81,0.19),\ \mathbf{r}^0=(0.11,0.89)$",
             fontsize=9.5, y=0.97)
fig.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.97])
save(fig, "fig_bif_coord")

Ps, Rs = coord_diagram([0.81, 0.19], [0.81, 0.19], AL)
Pm, Rm = coord_diagram([0.81, 0.19], [0.19, 0.81], AL)
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7))
for ax, (Pp, Rr), title in ((axs[0], (Ps, Rs), r"symmetric start $\mathbf{p}^0=\mathbf{r}^0$"),
                            (axs[1], (Pm, Rm), r"mirror start $p^0_1=r^0_2$")):
    ax.scatter(np.repeat(AL, Pp.shape[1]), Pp.ravel(), s=1.2, c=BLUE, lw=0, label=r"$p_1$")
    ax.scatter(np.repeat(AL, Rr.shape[1])[::3], Rr.ravel()[::3], s=3, marker="x",
               c=RED, lw=0.4, label=r"$r_1$")
    coord_axes(ax, r"$p_1,\ r_1$", title)
axs[0].legend(loc="lower right", framealpha=0.95, markerscale=3)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_bif_coord_sym_mirror")

# ================================================================ trajectories
fig, axs = plt.subplots(1, 2, figsize=(W, 3.25))
a = 1.5; xc, yc = cycle4(a); ax = axs[0]
for k, (dx, dy) in enumerate([(0.03, 0.018), (-0.03, 0.015), (0.025, -0.012)]):
    x, y = xc + dx, yc + dy
    xs, ys = [x], [y]
    for _ in range(25):
        x, y = Phi(*Phi(x, y, a), a); xs.append(x); ys.append(y)
    col = (TEAL, BLUE, RED)[k]
    ax.plot(xs, ys, "-", c=col, lw=0.6, alpha=0.8)
    ax.plot(xs, ys, "o", ms=3, c=col, mew=0)
    ax.plot(xs[0], ys[0], "s", ms=5, mfc="white", mec="k", mew=0.8)
ax.plot(xc, yc, "*", ms=11, c="k")
ax.set_xlim(xc - 0.045, xc + 0.045); ax.set_ylim(yc - 0.03, yc + 0.03)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_title(r"$\alpha=1.5$: iterates of $\Phi^2$ near $(x^*,y^*)$")
grid(ax)
a = 2.5; ax = axs[1]; u, v = fixed_asym(a)
for (x, y) in [(0.95, 0.60), (0.62, 0.97), (0.80, 0.55), (0.56, 0.70), (0.70, 0.66), (0.66, 0.71)]:
    xs, ys = [x], [y]
    for _ in range(30):
        x, y = Phi(x, y, a); xs.append(x); ys.append(y)
    col = BLUE if xs[0] > ys[0] else RED
    ax.plot(xs, ys, "-", c=col, lw=0.8)
    ax.plot(xs, ys, "o", ms=2.5, c=col, mew=0)
    ax.plot(xs[0], ys[0], "s", ms=5, mfc="white", mec="k", mew=0.8)
ax.plot([u, v], [v, u], "*", ms=11, c="k", ls="none")
ax.plot(xsym(a), xsym(a), "D", ms=5, mfc="white", mec="k", mew=0.8)
ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.9)
ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1); ax.set_aspect("equal")
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_title(r"$\alpha=2.5$: iterates of $\Phi$")
grid(ax)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_trajectories")

# ================================================================ time series
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
for ax, a, title in ((axs[0], 1.5, r"$\alpha=1.5$: exchange of values (4-cycle)"),
                     (axs[1], 2.5, r"$\alpha=2.5$: winner and loser")):
    x, y = 0.80, 0.62
    xs, ys = [x], [y]
    for _ in range(40):
        x, y = Phi(x, y, a); xs.append(x); ys.append(y)
    ax.plot(xs, "-o", ms=3, lw=1.0, c=BLUE, label=r"$x^t=p^t_{\max}$")
    ax.plot(ys, "-s", ms=3, lw=1.0, c=RED, label=r"$y^t=r^t_{\max}$")
    ax.set_xlabel(r"$t$"); ax.set_ylim(0.5, 1.0); ax.set_title(title)
    grid(ax)
axs[0].legend(loc="upper right", framealpha=0.95)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_time_series")

# ================================================================ collapse at alpha = 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
rng_c = np.random.default_rng(7)
TT = 120; t = np.arange(TT)
styles = ["-", "--", "-.", ":", "-"]
for k in range(5):
    x, y = rng_c.uniform(0.5, 1, 2)
    G = []
    for _ in range(TT):
        G.append(max(x, y) - 0.5); x, y = Phi(x, y, 2.0)
    G = np.array(G)
    a1.semilogy(t, G, styles[k], c=PURPLE, lw=1.1, alpha=0.85)
    a2.plot(t[:-2], G[2:] / G[:-2], styles[k], c=PURPLE, lw=1.1, alpha=0.85)
a1.semilogy(t, 0.5 * 0.75 ** (t / 2), "-", c=GREY, lw=0.9)
a1.annotate(r"$\propto(3/4)^{t/2}$", xy=(80, 0.5 * 0.75 ** 40), xytext=(86, 3e-4),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
a1.set_xlabel(r"$t$"); a1.set_ylabel(r"$G_t-\frac{1}{2}$")
a1.set_title(r"$\alpha=2$: geometric collapse to $(\frac{1}{2},\frac{1}{2})$")
grid(a1)
a2.axhline(0.75, color="k", lw=0.7, ls=":")
a2.text(78, 0.77, r"$3/4$", fontsize=8.5)
a2.set_ylim(0.3, 1.02)
a2.set_xlabel(r"$t$"); a2.set_ylabel(r"$(G_{t+2}-\frac{1}{2})\,/\,(G_t-\frac{1}{2})$")
a2.set_title("Ratio over two steps")
grid(a2)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_collapse")


# ================================================================ basins
def iterate_grid(a, n=400, steps=4000):
    gx = np.linspace(0.5, 1, n + 2)[1:-1]
    gy = np.linspace(0.5, 1, n + 1)[1:]
    gy = gy[gy < 1]                       # offset grid: avoids the diagonal
    X, Y = np.meshgrid(gx, gy)
    x, y = X.copy(), Y.copy()
    for _ in range(steps):
        x, y = Phi(x, y, a)
    return X, Y, x, y


cmap = ListedColormap(["#aed6f1", "#f5b7b1", "#d5d8dc"])   # light blue, light red, grey
fig, axs = plt.subplots(1, 3, figsize=(W, 2.55))
for ax, a in zip(axs, (0.5, 1.5, 2.5)):
    X, Y, x, y = iterate_grid(a)
    u, v = cycle4(a) if a < 2 else fixed_asym(a)
    cls = np.full(X.shape, 2)             # grey: not at the asymmetric attractor
    cls[(abs(x - u) < 1e-6) & (abs(y - v) < 1e-6)] = 0
    cls[(abs(x - v) < 1e-6) & (abs(y - u) < 1e-6)] = 1
    # the PDF embeds exactly the computed 400 x 399 array (no resampling)
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=2.5, origin="lower", interpolation="none",
              extent=(X[0, 0] - dx / 2, X[0, -1] + dx / 2, Y[0, 0] - dy / 2, Y[-1, 0] + dy / 2),
              aspect="auto")
    ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.8)
    ax.plot(u, v, "*", ms=10, c=BLUE, mec="k", mew=0.5)
    ax.plot(v, u, "*", ms=10, c=RED, mec="k", mew=0.5)
    ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1); ax.set_aspect("equal")
    ax.set_xticks([0.5, 0.75, 1]); ax.set_yticks([0.5, 0.75, 1])
    ax.set_xlabel(r"$x^0$"); ax.set_ylabel(r"$y^0$")
    ax.set_title(r"$\alpha=%.1f$" % a)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_basins")

print("figures written to", OUT)