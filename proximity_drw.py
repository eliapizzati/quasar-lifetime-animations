"""
Proximity zone for a flickering (DRW) quasar. Same physics and format as proximity_tq.py.

The quasar has been shining for T_PRE before the clip starts (so the zone and the
He III heating front are established); during the clip L(t) follows a damped random
walk in log L. Each cell equilibrates with
    dy/dt = Gamma_UVB [ h(T) - (1 + q L/<L>) y ],  q = (R_S/r)^2
so R_p tracks a lagged, smoothed copy of L(t) (lag ~ t_eq ~ 1e4 yr near R_p).
He II heating: front radius ~ (emitted HeII-ionizing photons)^(1/3) ~ (int L dt)^(1/3).

  python proximity_drw.py        -> proximity_zone_drw.mp4
  python proximity_drw.py 10     -> still at t = 10 s
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d

# ---------------- knobs ----------------
SEED, SEED_LC = 3, 5
FPS, DUR = 30, 15.0
WINDOW = 600.0              # kyr shown during the clip
T_PRE = 3000.0              # kyr of prior quasar activity (burn-in)
DT_PRE = 2.0                # kyr, burn-in step
SIG_DEX, TAU_DRW = 0.35, 100.0   # DRW scatter [dex] and damping time [kyr]
T_BKG = 300.0
R_S = 16.0
RMAX = 10.0
TAU0 = 60.0
XHI_BKG = 1e-4
T0, DT_HEAT = 1.0e4, 3.0e4
R_HE_10MYR = 10.0           # He III front radius after 1e4 kyr of emission at <L>
W_HE = 0.35
N_LOS = 150

BG, FG = "#3d3d3d", "#ededed"
STAR, BLUE = "#f7d64a", "#9ec5ff"
WHITE, SPINE = "#ffffff", "#d8d8d8"
LAB, TICK = 21, 17
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "text.color": WHITE, "axes.labelcolor": WHITE,
                     "xtick.color": WHITE, "ytick.color": WHITE,
                     "xtick.labelsize": TICK, "ytick.labelsize": TICK})

NFR = int(FPS * DUR)
t_kyr = np.linspace(0, WINDOW, NFR)
DTF = t_kyr[1] - t_kyr[0]

# ---------------- lightcurve: DRW in log L over burn-in + clip ----------------
rl = np.random.default_rng(SEED_LC)
t_all = np.concatenate([np.arange(-T_PRE, 0, DT_PRE), t_kyr])
dt_all = np.diff(np.concatenate([[t_all[0] - DT_PRE], t_all]))
x = rl.normal(); xs = np.empty(len(t_all))
for j, d in enumerate(dt_all):
    a = np.exp(-d / TAU_DRW)
    x = a * x + np.sqrt(1 - a * a) * rl.normal(); xs[j] = x
L_all = 10 ** (SIG_DEX * xs)
L_all /= np.mean(L_all)
cumL = np.cumsum(L_all * dt_all)                   # emitted energy (in <L> kyr)
L = L_all[-NFR:]

# ---------------- gas ----------------
rng = np.random.default_rng(SEED)
NR = 1000
r = np.linspace(0.02, RMAX, NR)
dr = r[1] - r[0]
g = gaussian_filter1d(rng.normal(size=(N_LOS, NR)), 6, axis=1)
g /= g.std(axis=1, keepdims=True)
Delta = np.exp(0.7 * g - 0.5 * 0.49)
box = int(1.0 / dr)
q = (R_S / r) ** 2

def h_of(cum):
    Rh = R_HE_10MYR * (cum / 1e4) ** (1 / 3)
    front = 0.5 * (1 - np.tanh((r - Rh) / W_HE))
    return ((T0 + DT_HEAT * front) / T0) ** -0.7

def rp_all(y):
    F = np.exp(-TAU0 * Delta ** 2 * y)
    Fs = uniform_filter1d(F, box, axis=1)
    below = Fs < 0.1
    idx = np.where(below.any(1), below.argmax(1), NR - 1)
    return r[idx], F, Fs

# ---------------- evolve ----------------
# y(r, t) does not depend on density, so one profile serves every sightline;
# only the transmission (through Delta) differs between them.
y = np.ones(NR)
RP = np.empty((NFR, N_LOS)); YS = np.empty((NFR, NR))
reb = np.zeros(N_LOS)                                # strongest smoothed-flux rebound past R_p
nb = len(t_all) - NFR
for j in range(len(t_all)):
    h = h_of(cumL[j]); qq = q * L_all[j]
    yeq = h / (1 + qq)
    y = yeq + (y - yeq) * np.exp(-(1 + qq) / T_BKG * dt_all[j])
    if j >= nb:
        i = j - nb
        RP[i], _, Fs = rp_all(y)
        YS[i] = y
        reb = np.maximum(reb, np.where(r[None, :] > RP[i][:, None] + 0.3, Fs, 0).max(axis=1))
med = np.median(RP, axis=1)
lo16, hi84 = np.percentile(RP, 16, axis=1), np.percentile(RP, 84, axis=1)

# displayed sightline: tracks the median, no strong rebound past R_p
score = np.mean(np.abs(RP - med[:, None]), axis=0) + 8 * np.clip(reb - 0.12, 0, None)
k = int(np.argmin(score))
Yk = YS; Fk = np.exp(-TAU0 * Delta[k] ** 2 * Yk); RPk = RP[:, k]
Fks = uniform_filter1d(Fk, box, axis=1)
NY = 50
tex = gaussian_filter(rng.normal(size=(NY, NR)), (4, 6)); tex /= tex.std()
Delta2 = np.exp(0.7 * (0.75 * g[k][None, :] + 0.66 * tex) - 0.5 * 0.49)
logL = np.log10(L)

# ---------------- figure ----------------
fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
X0, W = 0.085, 0.825

def style(ax):
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    for s_ in ["bottom", "left"]:
        ax.spines[s_].set_color(SPINE); ax.spines[s_].set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5)

axS = fig.add_axes([X0, 0.80, W, 0.15])
cm_h = LinearSegmentedColormap.from_list("hi", ["#fbf6e3", "#c9d0e2", "#7f8db5"])
imH = axS.imshow(np.zeros((NY, NR)), extent=[0, RMAX, 0, 1], aspect="auto", cmap=cm_h,
                 vmin=-6.3, vmax=-3.5, origin="lower", interpolation="bilinear")
axS.set_xlim(-1.0, RMAX); axS.set_ylim(0, 1); axS.axis("off")
axS.add_patch(plt.Rectangle((0, 0), RMAX, 1, fill=False, ec=SPINE, lw=2.5))
cax = fig.add_axes([X0 + W + 0.012, 0.80, 0.011, 0.15])
cb = fig.colorbar(imH, cax=cax, ticks=[-6, -5, -4])
cb.outline.set_edgecolor(SPINE); cb.ax.tick_params(labelsize=15, colors=WHITE)
cb.set_label(r"$\log x_{\rm HI}$", fontsize=19, color=WHITE)
glow = axS.scatter([-0.5], [0.5], s=0, c=STAR, alpha=0.25, lw=0, zorder=4, clip_on=False)
qso = axS.scatter([-0.5], [0.5], s=500, marker="*", c=STAR, ec="#8a6d00", lw=0.8,
                  zorder=5, clip_on=False)

axF = fig.add_axes([X0, 0.50, W, 0.25], facecolor=BG); style(axF)
axF.set_xlim(-1.0, RMAX); axF.set_ylim(-0.05, 1.22)
axF.spines["bottom"].set_bounds(0, RMAX)
axF.set_xticks(np.arange(0, RMAX + 0.1, 2)); axF.set_yticks([0, 0.5, 1])
axF.set_xlabel("distance from quasar [pMpc]", fontsize=LAB, labelpad=3)
axF.set_ylabel(r"Ly$\alpha$ flux", fontsize=LAB)
axF.axhline(0.1, color=SPINE, lw=1.0, ls=":")
fline, = axF.plot([], [], color="#a9a9a9", lw=1.0)
sline, = axF.plot([], [], color=WHITE, lw=2.6)
ffill = [None]
rpl, = axF.plot([0, 0], [-0.05, 1.0], color=STAR, lw=2.6)
rpt = axF.text(0, 1.0, r"$R_{\rm p}$", color=STAR, fontsize=24, ha="left", va="bottom")

# lightcurve + R_p history, shared time axis
axL = fig.add_axes([X0, 0.255, W, 0.12], facecolor=BG); style(axL)
axL.set_xlim(0, WINDOW); axL.set_xticklabels([])
axL.set_ylim(logL.min() - 0.15, logL.max() + 0.15)
axL.set_yticks([]); axL.set_ylabel(r"$\log L$", fontsize=LAB)
lline, = axL.plot([], [], color=STAR, lw=2.6)
ldot, = axL.plot([], [], "o", color=STAR, ms=9)

axR = fig.add_axes([X0, 0.095, W, 0.15], facecolor=BG); style(axR)
axR.set_xlim(0, WINDOW)
axR.set_ylim(0.9 * lo16.min(), 1.08 * hi84.max())
axR.set_xlabel("time [kyr]", fontsize=LAB, labelpad=3)
axR.set_ylabel(r"$R_{\rm p}$ [pMpc]", fontsize=17)
axR.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
band = [None]
mline, = axR.plot([], [], color=BLUE, lw=3.4)
mdot, = axR.plot([], [], "o", color=BLUE, ms=9)

def frame(i):
    imH.set_data(np.log10(XHI_BKG * Delta2 * Yk[i][None, :]))
    l = L[i]
    qso.set_sizes([500 + 350 * min(l, 2.5)]); glow.set_sizes([6000 * min(l, 2.5)])
    fline.set_data(r, Fk[i]); sline.set_data(r, Fks[i])
    if ffill[0] is not None: ffill[0].remove()
    ffill[0] = axF.fill_between(r, 0, Fk[i], color=FG, alpha=0.10, lw=0)
    rpl.set_xdata([RPk[i]] * 2); rpt.set_x(RPk[i] + 0.12)
    m = slice(0, i + 1)
    lline.set_data(t_kyr[m], logL[m]); ldot.set_data([t_kyr[i]], [logL[i]])
    mline.set_data(t_kyr[m], med[m]); mdot.set_data([t_kyr[i]], [med[i]])
    if band[0] is not None: band[0].remove()
    band[0] = axR.fill_between(t_kyr[m], lo16[m], hi84[m], color=BLUE, alpha=0.18, lw=0)
    return []

if __name__ == "__main__":
    print(f"logL range {np.ptp(logL):.2f} dex | Rp med {med.min():.2f}-{med.max():.2f} | k={k}")
    if len(sys.argv) > 1:
        frame(min(max(int(float(sys.argv[1]) * FPS), 0), NFR - 1))   # t = DUR -> last frame
        fig.savefig("pz_drw_still.png", facecolor=BG)
    else:
        FuncAnimation(fig, frame, frames=NFR, blit=False).save(
            "proximity_zone_drw.mp4",
            writer=FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264",
                                extra_args=["-pix_fmt", "yuv420p"]),
            savefig_kwargs={"facecolor": BG})
