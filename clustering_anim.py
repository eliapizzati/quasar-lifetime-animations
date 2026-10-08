"""
Quasar clustering and the duty cycle.

Two boxes share the same clustered dark-matter halo field and show the same mean number
of active quasars. Left: quasars live only in the most massive (most strongly clustered)
haloes, so each host must be active most of the time (high duty cycle). Right: quasars
live in many more, less massive haloes, so each is active only rarely (low duty cycle).
Only the clustering strength tells the two apart.

Lightcurve modes (first argument):
  onoff   simple on/off episodes
  drw     damped random walk in log L; a host is a quasar while log L > threshold,
          with the threshold set so the time above it equals f_duty

Flags:
  --extreme  f_duty = 0.85 (13 hosts) vs 0.05 (220 hosts)  [default: 0.75 vs 0.15]
  --info     add N_QSO / N_host counters and f_duty under each box
  --full     full-slide version: titles, counters, xi(r) panel, equations, punchline
  --crop     also write a version cropped to the two boxes + lightcurves (needs ffmpeg)

Examples:
  python clustering_anim.py onoff --extreme --info --crop
  python clustering_anim.py drw --extreme --info --crop
  python clustering_anim.py drw 12 --extreme --info      # still frame at t = 12 s
  python clustering_anim.py 12                           # same, default mode (drw)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import norm

FULL = "--full" in sys.argv          # full slide: titles, counters, xi(r), equations
INFO = "--info" in sys.argv          # boxes + lightcurves + N_QSO/N_host/f_duty text only
EXTREME = "--extreme" in sys.argv    # f_duty 0.85 vs 0.05
CROP = "--crop" in sys.argv          # also write a cropped (boxes + lightcurves) video
sys.argv = [a for a in sys.argv if a not in ("--full", "--info", "--extreme", "--crop")]
MODE, STILL_T = "drw", None         # positionals: lightcurve mode and/or still time [s]
for _a in sys.argv[1:]:
    if _a in ("onoff", "drw"):
        MODE = _a
    else:
        try:
            STILL_T = float(_a)
        except ValueError:
            sys.exit(f"unknown argument {_a!r}: expected onoff|drw, a time in seconds, "
                     "or --extreme/--info/--full/--crop")
SEED_FIELD, SEED_A, SEED_B = 7, 45, 21
TRACK_A = 12      # host shown in the left lightcurve strip (DRW mode; None = auto-pick)

# ---------------- knobs ----------------
N_QSO = 12            # mean number of visible quasars (both boxes)
N_HOST_A = 16         # left: most massive haloes only   -> f_duty = 0.75
N_HOST_B = 80         # right: top-80 haloes by mass     -> f_duty = 0.15
TAU_A, TAU_B = 3.0, 1.2    # DRW damping times (animation seconds)
PER_A, PER_B = 6.0, 2.5    # on/off cycle periods (animation seconds)
NH = 210
if EXTREME:
    N_QSO, N_HOST_A, N_HOST_B = 11, 13, 220     # f_duty = 0.85 and 0.05
    PER_A, PER_B = 8.0, 6.0
    NH = 290

BG, FG = "#3d3d3d", "#ededed"
GREY, GREEN, STAR = "#b8b8b8", "#5fcf80", "#f7d64a"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "text.color": FG, "axes.labelcolor": FG})

FPS, DUR = 30, 20.0
NFR = int(FPS * DUR)
T_FIELD = (0.0, 1.5)
T_HOST = (2.0, 3.5)
T_QSO = 4.0
DT = 1.0 / FPS
t_frames = np.arange(NFR) * DT

def ease(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return x * x * (3 - 2 * x)

# ---------------- clustered halo field ----------------
rng = np.random.default_rng(SEED_FIELD)
NG = 256
k = np.fft.fftfreq(NG) * NG
kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2); kk[0, 0] = 1
pk = kk ** -2.8 * np.exp(-(kk / 25) ** 2); pk[0, 0] = 0
delta = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(NG, NG))) * np.sqrt(pk)))
delta /= delta.std()

logM = np.clip(np.sort(rng.pareto(1.6, NH) * 0.35)[::-1], 0, 1.6)   # sorted, massive first
rad = 0.009 + 0.032 * (logM / logM.max()) ** 1.2
bias = 0.3 + 4.0 * (logM / logM.max())

pos = np.zeros((NH, 2))
for i in range(NH):
    for _ in range(20000):
        p = rng.uniform(0.03, 0.97, 2)
        d = delta[int(p[0] * NG) % NG, int(p[1] * NG) % NG]
        if rng.uniform() > np.exp(bias[i] * (d - 2.5)):
            continue
        if i and np.min(np.hypot(*(pos[:i] - p).T) - rad[:i]) < rad[i] + 0.004:
            continue
        pos[i] = p
        break

hostsA = np.arange(N_HOST_A)
hostsB = np.arange(N_HOST_B)
fA, fB = N_QSO / N_HOST_A, N_QSO / N_HOST_B

# ---------------- lightcurves ----------------
rA_ = np.random.default_rng(SEED_A)
rB_ = np.random.default_rng(SEED_B)

def drw(rlc, n, tau, burn=40.0):
    nb = int(burn / DT)
    a = np.exp(-DT / tau); s = np.sqrt(1 - a * a)
    x = rlc.normal(size=n)
    out = np.empty((NFR, n))
    for j in range(nb + NFR):
        x = a * x + s * rlc.normal(size=n)
        if j >= nb:
            out[j - nb] = x
    return out

def onoff(rlc, n, per, f, soft=0.05):
    ph = (np.arange(n) + rlc.uniform(0, 0.3, n)) / n
    rlc.shuffle(ph)
    x = np.mod((t_frames[:, None] - T_QSO) / per + ph[None, :], 1.0)
    return 0.5 * (1 + np.tanh((f - x) / (soft / per)))   # ~1 on, ~0 off

if MODE == "drw":
    lA, lB = drw(rA_, N_HOST_A, TAU_A), drw(rB_, N_HOST_B, TAU_B)
    thrA, thrB = norm.ppf(1 - fA), norm.ppf(1 - fB)
else:
    lA, lB = onoff(rA_, N_HOST_A, PER_A, fA), onoff(rB_, N_HOST_B, PER_B, fB)
    thrA = thrB = 0.5

def pick_track(l, thr, want, min_len=0.4):
    """Host whose visible stretch shows its duty cycle cleanly: the right number of
    well-resolved episodes, spread over the window, close to the mean duty cycle,
    and no long-term drift."""
    v = l[t_frames >= T_QSO]
    on = v > thr
    n = len(v)
    best, best_s = 0, -np.inf
    for j in range(v.shape[1]):
        o = np.concatenate([[0], on[:, j].astype(int), [0]])
        st, en = np.where(np.diff(o) == 1)[0], np.where(np.diff(o) == -1)[0]
        long_ = (en - st) * DT >= min_len
        nb = long_.sum()
        short = (~long_).sum()
        first = st[long_][0] / n if nb else 1.0
        drift = abs(v[: n // 2, j].mean() - v[n // 2 :, j].mean())
        sc = (-abs(nb - want) - 0.5 * short - 10 * abs(on[:, j].mean() - on.mean())
              - 2 * drift - 3 * max(0, first - 0.3))
        if sc > best_s:
            best, best_s = j, sc
    return best

trackA = (TRACK_A if (MODE == "drw" and TRACK_A is not None and TRACK_A < N_HOST_A)
          else pick_track(lA, thrA, 2))
trackB = pick_track(lB, thrB, 2 if EXTREME else 4)

# ---------------- figure ----------------
fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
BOX_W = 0.30
def box_axes(x0):
    ax = fig.add_axes([x0, 0.30, BOX_W, BOX_W * 12.8 / 7.2], facecolor="#f1f1f1")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9a9a"); s.set_linewidth(4)
    return ax

X0A, X0B = 0.035, 0.375
axA, axB = box_axes(X0A), box_axes(X0B)
sizes = np.pi * (rad * BOX_W * 12.8 * 72) ** 2

def make_layer(ax, hosts):
    sc = ax.scatter(*pos.T, s=sizes, c=GREY, lw=0, zorder=1)
    st = ax.scatter(*pos[hosts].T, s=np.zeros(len(hosts)), marker="*", c=STAR,
                    edgecolors="#8a6d00", lw=0.6, zorder=3)
    return sc, st

scA, stA = make_layer(axA, hostsA)
scB, stB = make_layer(axB, hostsB)

def lc_axes(x0, l, thr, label_thr=False):
    ax = fig.add_axes([x0, 0.07, BOX_W, 0.13], facecolor=BG)
    v = l[t_frames >= T_QSO]
    if MODE == "drw":
        lo, hi = min(v.min(), thr) - 0.3, max(v.max(), thr) + 0.9
    else:
        lo, hi = -0.15, 1.35
    ax.set_xlim(T_QSO - 0.5, DUR); ax.set_ylim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ["top", "right", "left"]:
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color("#9a9a9a")
    ax.set_xlabel("time", fontsize=11, labelpad=2)
    ax.text(T_QSO - 0.4, hi - 0.12 * (hi - lo), "one host:", fontsize=10,
            color="#9ec5ff", va="center")
    if MODE == "drw":
        ax.axhline(thr, color="#cfcfcf", lw=1.0, ls="--")
        if label_thr:
          ax.text(DUR - 0.1, thr + 0.08 * (hi - lo), "threshold", ha="right", va="bottom",
                fontsize=10, color="#cfcfcf")
    line, = ax.plot([], [], color="#d9d9d9", lw=1.8)
    above, = ax.plot([], [], color=STAR, lw=2.2)
    dot, = ax.plot([], [], "o", ms=6, color=STAR)
    ax.set_visible(False)
    return dict(ax=ax, l=l, thr=thr, line=line, above=above, dot=dot, fill=None)

LCA = lc_axes(X0A, lA[:, trackA], thrA)
LCB = lc_axes(X0B, lB[:, trackB], thrB, label_thr=True)

grey = np.array(matplotlib.colors.to_rgba(GREY))
green = np.array(matplotlib.colors.to_rgba(GREEN))
ystar = np.array(matplotlib.colors.to_rgba(STAR))
yedge = np.array(matplotlib.colors.to_rgba("#8a6d00"))

def update_box(sc, st, hosts, l, thr, i, t):
    a_f, a_h = ease(t, *T_FIELD), ease(t, *T_HOST)
    cols = np.tile(grey, (NH, 1))
    cols[hosts] = (1 - a_h) * grey + a_h * green
    cols[:, 3] = a_f
    sc.set_facecolors(cols)
    if t < T_QSO:
        return
    x = l[i] - thr
    if MODE == "drw":
        a = np.clip(x / 0.25, 0, 1)
        s = 110 + 120 * np.clip(x, 0, 2)          # brighter -> bigger
    else:
        a = np.clip(l[i], 0, 1)
        s = np.full(len(hosts), 230.0)
    a = a * ease(t, T_QSO, T_QSO + 0.5)
    fc = np.tile(ystar, (len(hosts), 1)); fc[:, 3] = a
    ec = np.tile(yedge, (len(hosts), 1)); ec[:, 3] = a
    st.set_facecolors(fc); st.set_edgecolors(ec); st.set_sizes(s)

def update_lc(d, i, t):
    if t < T_QSO:
        return
    d["ax"].set_visible(True)
    m = (t_frames >= T_QSO - 0.5) & (t_frames <= t)
    tt, yy = t_frames[m], d["l"][m]
    d["line"].set_data(tt, yy)
    cut = d["thr"] if MODE == "drw" else 0.03
    d["above"].set_data(tt, np.where(yy > cut, yy, np.nan))
    d["dot"].set_data([t], [d["l"][i]])
    d["dot"].set_color(STAR if d["l"][i] > d["thr"] else "#d9d9d9")
    if MODE == "drw":
        if d["fill"] is not None:
            d["fill"].remove()
        d["fill"] = d["ax"].fill_between(tt, d["thr"], yy, where=yy > d["thr"],
                                         color=STAR, alpha=0.25, lw=0)

# ---------------- full-slide overlay ----------------
R0A, R0B, GAM = 11.0, 6.5, 1.8          # schematic r0 [h^-1 cMpc]
if EXTREME:
    R0B = 4.5                             # hosts reach much lower masses
T_XI = (11.0, 13.5)
T_EQ1, T_EQ2, T_EQ3, T_PUNCH = 13.5, 15.0, 16.5, 18.0
LIGHT = "#9be7b4"
if FULL:
    cA_, cB_ = X0A + BOX_W / 2, X0B + BOX_W / 2
    fig.text(cA_, 0.925, "Large halos, high duty cycle", ha="center", fontsize=17, weight="bold")
    fig.text(cA_, 0.88, "quasars stay on", ha="center", fontsize=13, color="#cfcfcf")
    fig.text(cB_, 0.925, "Small halos, low duty cycle", ha="center", fontsize=17, weight="bold")
    fig.text(cB_, 0.88, "quasars flicker", ha="center", fontsize=13, color="#cfcfcf")
    cntA = fig.text(cA_, 0.258, "", ha="center", fontsize=14, color=STAR)
    cntB = fig.text(cB_, 0.258, "", ha="center", fontsize=14, color=STAR)
    eq3A = fig.text(cA_, 0.215, rf"$f_{{\rm duty}} \approx {fA:.2f}$", ha="center",
                    fontsize=15, color=GREEN, alpha=0)
    eq3B = fig.text(cB_, 0.215, rf"$f_{{\rm duty}} \approx {fB:.2f}$", ha="center",
                    fontsize=15, color=LIGHT, alpha=0)

    axX = fig.add_axes([0.765, 0.50, 0.215, 0.36], facecolor=BG)
    rr = np.logspace(-0.3, 1.6, 200)
    axX.set_xscale("log"); axX.set_yscale("log")
    axX.set_xlim(rr[0], rr[-1]); axX.set_ylim(0.03, 300)
    for s_ in axX.spines.values():
        s_.set_color("#9a9a9a")
    axX.tick_params(labelsize=9, colors=FG)
    axX.set_xlabel(r"$r\ [h^{-1}\,\mathrm{cMpc}]$", fontsize=11)
    axX.set_ylabel(r"$\xi_{\rm QQ}(r)$", fontsize=13)
    axX.axhline(1, color="#8a8a8a", lw=1, ls=":")
    xiA, = axX.plot([], [], color=GREEN, lw=2.6, label="large halos")
    xiB, = axX.plot([], [], color=LIGHT, lw=2.6, ls="--", label="small halos")
    r0l = [axX.axvline(R0A, color=GREEN, lw=1, alpha=0),
           axX.axvline(R0B, color=LIGHT, lw=1, alpha=0)]
    r0t = [axX.text(R0A * 1.08, 0.05, r"$r_0$", color=GREEN, fontsize=12, alpha=0),
           axX.text(R0B * 1.08, 0.05, r"$r_0$", color=LIGHT, fontsize=12, alpha=0)]
    axX.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=FG)
    axX.set_visible(False)

    eq1 = fig.text(0.872, 0.385,
                   r"$r_0 \rightarrow b(M) \rightarrow M_{\rm host} \rightarrow n_{\rm host}$",
                   ha="center", fontsize=14, alpha=0)
    eq2 = fig.text(0.872, 0.285,
                   r"$f_{\rm duty} = \dfrac{n_{\rm QSO}}{n_{\rm host}} \sim \dfrac{t_{\rm Q}}{t_{\rm H}}$",
                   ha="center", fontsize=16, alpha=0)
    punch = fig.text(0.872, 0.13, "Same number of quasars.\nOnly clustering\ntells them apart.",
                     ha="center", fontsize=14, weight="bold", color=STAR, alpha=0,
                     linespacing=1.4)

if INFO and not FULL:
    cA_, cB_ = X0A + BOX_W / 2, X0B + BOX_W / 2
    cntA = fig.text(cA_, 0.25, "", ha="center", fontsize=17, color=STAR)
    cntB = fig.text(cB_, 0.25, "", ha="center", fontsize=17, color=STAR)
    eq3A = fig.text(cA_, 0.205, rf"$f_{{\rm duty}} = {fA:.2f}$", ha="center",
                    fontsize=18, color=GREEN, alpha=0)
    eq3B = fig.text(cB_, 0.205, rf"$f_{{\rm duty}} = {fB:.2f}$", ha="center",
                    fontsize=18, color=LIGHT, alpha=0)

def update_info(i, t):
    if t >= T_QSO:
        onA = int((lA[i] > thrA).sum()); onB = int((lB[i] > thrB).sum())
        cntA.set_text(rf"$N_{{\rm QSO}}$ = {onA:2d}   /   $N_{{\rm host}}$ = {N_HOST_A}")
        cntB.set_text(rf"$N_{{\rm QSO}}$ = {onB:2d}   /   $N_{{\rm host}}$ = {N_HOST_B}")
    a = ease(t, T_QSO + 1.0, T_QSO + 2.0); eq3A.set_alpha(a); eq3B.set_alpha(a)

def update_overlay(i, t):
    if t >= T_QSO:
        onA = int((lA[i] > thrA).sum()); onB = int((lB[i] > thrB).sum())
        cntA.set_text(rf"$N_{{\rm QSO}}$ on = {onA}   /   $N_{{\rm host}}$ = {N_HOST_A}")
        cntB.set_text(rf"$N_{{\rm QSO}}$ on = {onB}   /   $N_{{\rm host}}$ = {N_HOST_B}")
    if t >= T_XI[0]:
        axX.set_visible(True)
        n = max(2, int(ease(t, *T_XI) * len(rr)))
        xiA.set_data(rr[:n], (rr[:n] / R0A) ** -GAM)
        xiB.set_data(rr[:n], (rr[:n] / R0B) ** -GAM)
        a = ease(t, T_XI[1] - 0.6, T_XI[1] + 0.4)
        for o in r0l: o.set_alpha(0.8 * a)
        for o in r0t: o.set_alpha(a)
    eq1.set_alpha(ease(t, T_EQ1, T_EQ1 + 0.8))
    eq2.set_alpha(ease(t, T_EQ2, T_EQ2 + 0.8))
    a3 = ease(t, T_EQ3, T_EQ3 + 0.8); eq3A.set_alpha(a3); eq3B.set_alpha(a3)
    punch.set_alpha(ease(t, T_PUNCH, T_PUNCH + 0.8))

def frame(i):
    t = t_frames[i]
    update_box(scA, stA, hostsA, lA, thrA, i, t)
    update_box(scB, stB, hostsB, lB, thrB, i, t)
    update_lc(LCA, i, t); update_lc(LCB, i, t)
    if FULL:
        update_overlay(i, t)
    elif INFO:
        update_info(i, t)
    return []

if __name__ == "__main__":
    if STILL_T is not None:
        frame(min(max(int(STILL_T * FPS), 0), NFR - 1))   # t = DUR -> last frame
        fig.savefig(f"still_{MODE}{'_full' if FULL else ('_info' if INFO else '')}{'_extreme' if EXTREME else ''}.png", facecolor=BG)
    else:
        nA = (lA[t_frames >= T_QSO] > thrA).sum(1)
        nB = (lB[t_frames >= T_QSO] > thrB).sum(1)
        print(f"[{MODE}] f_duty A={fA:.2f} B={fB:.2f} | N_on A {nA.mean():.1f}±{nA.std():.1f}"
              f"  B {nB.mean():.1f}±{nB.std():.1f}")
        out = f"clustering_dutycycle_{MODE}{'_full' if FULL else ('_info' if INFO else '')}{'_extreme' if EXTREME else ''}.mp4"
        FuncAnimation(fig, frame, frames=NFR, blit=False).save(
            out,
            writer=FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264",
                                extra_args=["-pix_fmt", "yuv420p"]),
            savefig_kwargs={"facecolor": BG})
        if CROP and not FULL:
            import subprocess
            cropped = out.replace("clustering_dutycycle_", "clustering_panels_").replace("_info", "")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out,
                            "-vf", "crop=1320:930:20:150", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-b:v", "6M", cropped], check=True)
            print("wrote", cropped)
