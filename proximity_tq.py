"""
Proximity zone vs quasar lifetime t_Q (lightbulb: switches on once, stays on).

Physics (schematic, per cell along the sightline):
  H I:   dy/dt = Gamma_UVB [ h(T) - (1+q) y ],  q = (R_S/r)^2,  y = x_HI/x_HI,bkg
         equilibrium y_eq = h(T)/(1+q),  t_eq = 1/[Gamma_UVB (1+q)]
  He II: background helium is mostly He II at z~6, so the quasar's hard photons drive
         an ionization front that expands by photon counting, R_HeIII ~ t_Q^(1/3).
         Behind it the gas is photoheated by dT_HEAT; alpha_A ~ T^-0.7 lowers x_HI:
         h(T) = (T/T0)^-0.7
  Lya:   tau = TAU0 Delta^2 y  (fixed T-dependence of tau beyond alpha is ignored),
         R_p = first r where the ~1 pMpc-smoothed transmission drops below 0.1.
Bottom panel: median R_p over N_LOS sightlines with its 16-84% band. The spectrum panel
shows one sightline: raw transmission (thin grey) and the smoothed flux (white) used to
define R_p. Parameters are illustrative, tuned for a slide, not fit to data.

  python proximity_tq.py        -> proximity_zone_tq.mp4
  python proximity_tq.py 15     -> still at t = 15 s
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
SEED = 3
FPS, DUR = 30, 15.0
T_ON = 1.0                  # switch-on (animation s)
TQ_RANGE = (0.1, 1e4)       # quasar age shown [kyr]: 1e2 -> 1e7 yr, log-spaced
TQ_HOLD = 1.5               # seconds held on last frame
T_BKG = 300.0               # 1/Gamma_UVB [kyr]
R_S = 16.0                  # r where Gamma_QSO = Gamma_UVB [pMpc]
RMAX = 10.0                 # pMpc shown
TAU0 = 60.0
XHI_BKG = 1e-4
T0, DT_HEAT = 1.0e4, 3.0e4  # IGM temperature and He II photoheating [K]
R_HE_10MYR = 10.0            # He III front radius at t_Q = 10 Myr [pMpc]; R ~ t^(1/3)
W_HE = 0.35                 # front width [pMpc]
N_LOS = 150

BG, FG = "#3d3d3d", "#ededed"
STAR, BLUE = "#f7d64a", "#9ec5ff"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG})

NFR = int(FPS * DUR)
t_s = np.arange(NFR) / FPS
u = np.clip((t_s - T_ON) / (DUR - TQ_HOLD - T_ON), 0, 1)
tQ = np.where(t_s >= T_ON, TQ_RANGE[0] * (TQ_RANGE[1] / TQ_RANGE[0]) ** u, 0.0)
dts = np.diff(np.concatenate([[0.0], tQ]))

# ---------------- density fields ----------------
rng = np.random.default_rng(SEED)
NR = 1000
r = np.linspace(0.02, RMAX, NR)
dr = r[1] - r[0]
g = gaussian_filter1d(rng.normal(size=(N_LOS, NR)), 6, axis=1)
g /= g.std(axis=1, keepdims=True)
Delta = np.exp(0.7 * g - 0.5 * 0.49)
box = int(1.0 / dr)

def rp_all(y):
    F = np.exp(-TAU0 * Delta ** 2 * y)
    Fs = uniform_filter1d(F, box, axis=1)
    below = Fs < 0.1
    idx = np.where(below.any(1), below.argmax(1), NR - 1)
    return r[idx], F

# ---------------- He III front / heating ----------------
def R_he(tq_kyr):
    return R_HE_10MYR * (np.maximum(tq_kyr, 0) / 1e4) ** (1 / 3)

def temp(tq_kyr):
    front = 0.5 * (1 - np.tanh((r - R_he(tq_kyr)) / W_HE))
    return T0 + DT_HEAT * front * (tq_kyr > 0)

# ---------------- evolve all sightlines ----------------
q = (R_S / r) ** 2
y = np.ones((N_LOS, NR))
RP = np.zeros((NFR, N_LOS))
TT = np.empty((NFR, NR))
for i in range(NFR):
    T = temp(tQ[i]); TT[i] = T
    h = (T / T0) ** -0.7
    if t_s[i] >= T_ON:
        yeq = h / (1 + q)
        y = yeq + (y - yeq) * np.exp(-(1 + q) / T_BKG * dts[i])
    RP[i], F = rp_all(y)
RP[t_s < T_ON] = 0.0

# displayed sightline: background R_p small, final R_p close to the median
med = np.median(RP, axis=1)
bg_rp = rp_all(np.ones((N_LOS, NR)))[0]
score = np.abs(RP[-1] - med[-1]) + 3 * (bg_rp > 0.3) + \
        np.mean(np.abs(RP[t_s >= T_ON] - med[t_s >= T_ON, None]), axis=0)
# prefer sightlines whose smoothed flux stays below 0.1 beyond R_p at late times
# (R_p is the FIRST crossing; strong rebounds beyond it look misleading on a slide)
Ffin = np.exp(-TAU0 * Delta ** 2 * y)              # y = final state of all sightlines
Fsfin = uniform_filter1d(Ffin, box, axis=1)
rebound = np.array([Fsfin[j, r > RP[-1, j] + 0.3].max() if (r > RP[-1, j] + 0.3).any()
                    else 0 for j in range(N_LOS)])
score = score + 8 * np.clip(rebound - 0.1, 0, None)
k = int(np.argmin(score))
print("chosen sightline", k, "rebound", round(rebound[k], 3))
# recompute the chosen sightline's fields for display
yk = np.ones(NR); Yk = np.empty((NFR, NR))
for i in range(NFR):
    h = (TT[i] / T0) ** -0.7
    if t_s[i] >= T_ON:
        yeq = h / (1 + q)
        yk = yeq + (yk - yeq) * np.exp(-(1 + q) / T_BKG * dts[i])
    Yk[i] = yk
Fk = np.exp(-TAU0 * Delta[k] ** 2 * Yk)
RPk = RP[:, k]
NY = 50
tex = gaussian_filter(rng.normal(size=(NY, NR)), (4, 6)); tex /= tex.std()
Delta2 = np.exp(0.7 * (0.75 * g[k][None, :] + 0.66 * tex) - 0.5 * 0.49)


# ---------------- figure ----------------
Fks = uniform_filter1d(Fk, box, axis=1)        # ~1 pMpc boxcar: R_p is defined on this
WHITE, SPINE = "#ffffff", "#d8d8d8"
LAB, TICK = 21, 17
plt.rcParams.update({"axes.labelcolor": WHITE, "xtick.color": WHITE, "ytick.color": WHITE,
                     "xtick.labelsize": TICK, "ytick.labelsize": TICK})
fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
X0, W = 0.085, 0.825

def style(ax):
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    for s_ in ["bottom", "left"]:
        ax.spines[s_].set_color(SPINE); ax.spines[s_].set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5)

# (1) sightline strip
axS = fig.add_axes([X0, 0.785, W, 0.165])
cm_h = LinearSegmentedColormap.from_list("hi", ["#fbf6e3", "#c9d0e2", "#7f8db5"])
imH = axS.imshow(np.zeros((NY, NR)), extent=[0, RMAX, 0, 1], aspect="auto", cmap=cm_h,
                 vmin=-6.3, vmax=-3.5, origin="lower", interpolation="bilinear")
axS.set_xlim(-1.0, RMAX); axS.set_ylim(0, 1); axS.axis("off")
axS.add_patch(plt.Rectangle((0, 0), RMAX, 1, fill=False, ec=SPINE, lw=2.5))
cax = fig.add_axes([X0 + W + 0.012, 0.785, 0.011, 0.165])
cb = fig.colorbar(imH, cax=cax, ticks=[-6, -5, -4])
cb.outline.set_edgecolor(SPINE); cb.ax.tick_params(labelsize=15, colors=WHITE)
cb.set_label(r"$\log x_{\rm HI}$", fontsize=19, color=WHITE)
glow = axS.scatter([-0.5], [0.5], s=0, c=STAR, alpha=0.25, lw=0, zorder=4, clip_on=False)
qso = axS.scatter([-0.5], [0.5], s=500, marker="*", c=STAR, ec="#8a6d00", lw=0.8,
                  zorder=5, clip_on=False)

# (2) transmission
axF = fig.add_axes([X0, 0.475, W, 0.27], facecolor=BG); style(axF)
axF.set_xlim(-1.0, RMAX); axF.set_ylim(-0.05, 1.22)
axF.spines["bottom"].set_bounds(0, RMAX)
axF.set_xticks(np.arange(0, RMAX + 0.1, 2)); axF.set_yticks([0, 0.5, 1])
axF.set_xlabel(r"distance from quasar [pMpc]", fontsize=LAB, labelpad=3)
axF.set_ylabel(r"Ly$\alpha$ flux", fontsize=LAB)
axF.axhline(0.1, color=SPINE, lw=1.0, ls=":")
fline, = axF.plot([], [], color="#a9a9a9", lw=1.0)
sline, = axF.plot([], [], color=WHITE, lw=2.6)
ffill = [None]
rpl, = axF.plot([0, 0], [-0.05, 1.0], color=STAR, lw=2.6, alpha=0)
rpt = axF.text(0, 1.0, r"$R_{\rm p}$", color=STAR, fontsize=24, ha="left", va="bottom", alpha=0)

# (3) R_p - t_Q relation
axR = fig.add_axes([X0, 0.105, W, 0.27], facecolor=BG); style(axR)
axR.set_xscale("log"); axR.set_xlim(*TQ_RANGE)
lo16, hi84 = np.percentile(RP, 16, axis=1), np.percentile(RP, 84, axis=1)
YMAX = 1.15 * hi84.max()
axR.set_ylim(0, YMAX); axR.set_yticks([0, 2, 4, 6, 8])
axR.set_xticks([0.1, 1, 10, 100, 1e3, 1e4])
axR.set_xticklabels([rf"$10^{e}$" for e in range(2, 8)])
axR.minorticks_off()
axR.set_xlabel(r"quasar lifetime $t_{\rm Q}$ [yr]", fontsize=LAB, labelpad=3)
axR.set_ylabel(r"$R_{\rm p}$ [pMpc]", fontsize=LAB)
band = [None]
mline, = axR.plot([], [], color=BLUE, lw=3.6)
mdot, = axR.plot([], [], "o", color=BLUE, ms=9)
tq_txt = axR.text(0.012, 0.97, "", transform=axR.transAxes, ha="left", va="top",
                 fontsize=24, color=STAR)

def fmt_tq(kyr):
    """Plain number, 2 significant figures, thousands separators: no moving exponent."""
    yr = kyr * 1e3
    e = int(np.floor(np.log10(yr)))
    v = int(round(yr / 10 ** (e - 1)) * 10 ** (e - 1))
    return rf"$t_{{\rm Q}}$ = {v:,} yr"

def frame(i):
    t, tq = t_s[i], tQ[i]
    imH.set_data(np.log10(XHI_BKG * Delta2 * Yk[i][None, :]))
    on = t >= T_ON
    qso.set_sizes([800 if on else 250]); qso.set_alpha(1.0 if on else 0.25)
    glow.set_sizes([9000 if on else 0])
    fline.set_data(r, Fk[i]); sline.set_data(r, Fks[i])
    if ffill[0] is not None: ffill[0].remove()
    ffill[0] = axF.fill_between(r, 0, Fk[i], color=FG, alpha=0.10, lw=0)
    a = 1.0 if on and RPk[i] > 0.3 else 0.0
    rpl.set_xdata([RPk[i]] * 2); rpl.set_alpha(a); rpt.set_x(RPk[i] + 0.12); rpt.set_alpha(a)
    if on:
        m = (t_s <= t) & (t_s >= T_ON)
        mline.set_data(tQ[m], med[m]); mdot.set_data([tq], [med[i]])
        if band[0] is not None: band[0].remove()
        band[0] = axR.fill_between(tQ[m], lo16[m], hi84[m], color=BLUE, alpha=0.18, lw=0)
        tq_txt.set_text(fmt_tq(tq))
    return []

if __name__ == "__main__":
    for x in [0.3, 1, 3, 10, 30, 100, 1e3, 3e3, 1e4]:
        j = np.argmin(abs(tQ - x)); print(f"  tQ={x*1e3:.0e} yr  Rp_med={med[j]:.2f}  Rp_k={RPk[j]:.2f}")
    if len(sys.argv) > 1:
        frame(int(float(sys.argv[1]) * FPS)); fig.savefig("pz_tq_still.png", facecolor=BG)
    else:
        FuncAnimation(fig, frame, frames=NFR, blit=False).save(
            "proximity_zone_tq.mp4",
            writer=FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264",
                                extra_args=["-pix_fmt", "yuv420p"]),
            savefig_kwargs={"facecolor": BG})
