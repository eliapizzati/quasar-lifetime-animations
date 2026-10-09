# Quasar lifetime animations

Animations for talks on how two observables constrain quasar lightcurves:

- **Clustering** fixes the host-halo mass, hence the host abundance, hence the
  **duty cycle** $f_{\rm duty} = n_{\rm QSO}/n_{\rm host}$, the fraction of time a host is active.
- **Proximity zones** respond to the quasar's ionizing output on the H I equilibration time
  ($\sim 10^4$ yr), so their size $R_{\rm p}$ probes the **episodic** lifetime $t_{\rm Q}$
  and short-timescale variability.

Pedagogical toy models, tuned for clarity on a slide. Not fits to data.

## Videos

| File | What it shows |
|---|---|
| [`videos/clustering_panels_onoff_extreme.mp4`](videos/clustering_panels_onoff_extreme.mp4) | Same halo field, same number of visible quasars. Left: 13 massive hosts, $f_{\rm duty}=0.85$. Right: 220 hosts, $f_{\rm duty}=0.05$. On/off lightcurves. |
| [`videos/clustering_panels_drw_extreme.mp4`](videos/clustering_panels_drw_extreme.mp4) | As above with damped-random-walk lightcurves; a host is a quasar while $\log L$ is above a threshold. |
| [`videos/clustering_dutycycle_onoff_full_extreme.mp4`](videos/clustering_dutycycle_onoff_full_extreme.mp4) | Full-slide version of the on/off clip: titles, counters, schematic $\xi(r)$, the $r_0 \to b(M) \to M_{\rm host} \to n_{\rm host}$ chain and $f_{\rm duty} = n_{\rm QSO}/n_{\rm host} \sim t_{\rm Q}/t_{\rm H}$. |
| [`videos/clustering_dutycycle_drw_full_extreme.mp4`](videos/clustering_dutycycle_drw_full_extreme.mp4) | Full-slide version with DRW lightcurves. |
| [`videos/clustering_panels_drw_edd.mp4`](videos/clustering_panels_drw_edd.mp4) | Same duty cycles and one common survey limit in $L_{\rm bol}$: the duty cycles differ because massive hosts sit above the limit on average and small hosts below it. Both lightcurves are DRWs differing only in coherence time. Left: long $\tau_{\rm DRW}$, long sustained near-Eddington episodes with rare drops below the limit. Right: short $\tau_{\rm DRW}$, brief flares above the limit. In both, the mean luminosity (dotted line) rises linearly in $\log L$, mimicking exponential Eddington-limited BH growth ($L \propto M_{\rm BH} \propto e^{t/t_{\rm Sal}}$); with a fixed survey limit, more quasars are visible at the end than at the start, so the duty cycles shown are clip averages $\langle f_{\rm duty}\rangle$. |
| [`videos/clustering_dutycycle_drw_full_edd.mp4`](videos/clustering_dutycycle_drw_full_edd.mp4) | Full-slide version of the above. |
| [`videos/clustering_panels_drw_bulb.mp4`](videos/clustering_panels_drw_bulb.mp4) | Same shared survey limit and duty cycles. Static (no growth trend). Left: toy lightbulb, few massive hosts that switch on and stay on at constant $L$ for long episodes. Right: many small hosts with a bursty short-$\tau_{\rm DRW}$ lightcurve, brief flares above the limit. |
| [`videos/clustering_dutycycle_drw_full_bulb.mp4`](videos/clustering_dutycycle_drw_full_bulb.mp4) | Full-slide version of the above. |
| [`videos/clustering_panels_drw_bulb_light.mp4`](videos/clustering_panels_drw_bulb_light.mp4), [`videos/clustering_dutycycle_drw_full_bulb_light.mp4`](videos/clustering_dutycycle_drw_full_bulb_light.mp4) | The same two clips on a white background (`--light`). |
| [`videos/proximity_zone_tq.mp4`](videos/proximity_zone_tq.mp4) | Quasar switches on and stays on. $R_{\rm p}$ grows with $t_{\rm Q}$ until H I equilibrium (~$10^5$ yr), plateaus, then grows again as the He II photoheating front passes. |
| [`videos/proximity_zone_drw.mp4`](videos/proximity_zone_drw.mp4) | Flickering (DRW) quasar: $R_{\rm p}$ follows a lagged, smoothed copy of $L(t)$. |

## Running

Requires Python 3 with `numpy`, `scipy`, `matplotlib`, and `ffmpeg` on your PATH.

```bash
pip install -r requirements.txt
./make_all.sh                      # renders everything (~1 min per video)
```

Individual scripts:

```bash
python clustering_anim.py onoff --extreme --info --crop   # or: drw
python clustering_anim.py onoff --extreme --full          # full-slide version with xi(r) and equations
python proximity_tq.py
python proximity_drw.py
python proximity_tq.py 10          # any script + a time in seconds -> still PNG of that frame
```

Each script lists its options in its docstring. All physical and visual parameters
(duty cycles, host numbers, DRW damping times, $\Gamma_{\rm UVB}$, heating amplitude,
He III front speed, …) are knobs at the top of the file.

## Model notes

**Clustering** (`clustering_anim.py`). Haloes are drawn from a Gaussian random field with
mass-dependent bias, so massive haloes cluster. Hosts are the top-$N$ haloes by mass
(the cap on halo size never falls below the least massive left-box host, so no non-host
is drawn as large as a host there);
$N_{\rm host}$ is chosen so that $f_{\rm duty} N_{\rm host}$ is the same in both boxes.
The $\xi(r)$ panel in `--full` mode is a schematic power law.

**Proximity zones** (`proximity_tq.py`, `proximity_drw.py`). Each cell along the sightline
relaxes toward photoionization equilibrium,

$$\frac{dy}{dt} = \Gamma_{\rm UVB}\left[h(T) - \left(1 + \frac{L}{\langle L\rangle}\frac{R_S^2}{r^2}\right) y\right],
\qquad y = x_{\rm HI}/x_{\rm HI,bkg},$$

so the equilibration time grows as $r^2$. He II photoheating: the He III front expands as
$R \propto (\int L\,dt)^{1/3}$ and heats the gas behind it, lowering $x_{\rm HI}$ via
$\alpha_A \propto T^{-0.7}$. Ly$\alpha$ optical depth $\tau \propto \Delta^2 y$ on a lognormal
density field. $R_{\rm p}$ is the first radius where the flux, smoothed over ~1 pMpc,
drops below 10%. Curves show the median over 150 sightlines with the 16–84% band.
Light-travel-time effects are ignored.

In the DRW clip the He III front is placed beyond the strip (`T_HE_PRIOR`: He III
recombines slowly, so the front remembers all past activity), so the gas shown is uniformly
heated and $R_{\rm p}$ responds to $L(t)$ alone. With the front inside the strip, $R_{\rm p}$
saturates at it whenever $L$ is high. The strip there is 14 pMpc long because the heated
zone reaches ~10 pMpc in the bright state.

## License

MIT, see [LICENSE](LICENSE). If you reuse the animations in a talk, a credit line is appreciated.
