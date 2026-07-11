"""Multi-arm full radial power-spectra OVERLAY (all wavenumbers) — testbed-validation figure.

Overlays each arm's 1-D radial power spectrum (from <arm>_physrealism.json spectrum_1d) for
10u/10v/msl against a shared truth + input, log-log vs wavelength, 20-100 km fine band shaded.
Current arms: b785(unified 375k) / eecdb127(ds-API 200k) / warmstart(regional 100k) /
scratch512(from-scratch 100k). Reconstructs the identical 347x747 / 0.075deg / 39-log-bin
wavenumber axis so it aligns with the saved spectra. Companion to render.py (single-run version).
Edit the ARMS dict to point at any set of *_physrealism.json.
"""
import json, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PERM="/home/ecm5702/perm/eval-rescue-20260709/regional_three_curve_verdict/physical_realism_20260710"
SCR="/home/ecm5702/scratch/eval/testbed_validation_plots"
arms={"b785matched":("b785 (unified 375k)","#8e44ad",f"{PERM}/b785matched_physrealism.json"),
      "eecdb127":("eecdb127 (ds-API 200k)","#e67e22",f"{SCR}/eecdb127_physrealism.json"),
      "warmstart":("warmstart (regional 100k)","#1a73e8",f"{PERM}/warmstart_physrealism.json"),
      "scratch512":("scratch512 (from-scratch 100k)","#c0392b",f"{PERM}/scratch512_physrealism.json")}
J={a:json.load(open(p)) for a,(lab,c,p) in arms.items()}
GRID=0.075; RIM=2.0
glat=np.arange(5+RIM,35-RIM+1e-9,GRID); glon=np.arange(-100+RIM,-40-RIM+1e-9,GRID)
ny,nx=len(glat),len(glon); DY=GRID*111.195; DX=DY*np.cos(np.deg2rad(20.0))
kk=np.sqrt(np.fft.fftfreq(ny,DY)[:,None]**2+np.fft.rfftfreq(nx,DX)[None,:]**2)
kmid=np.sqrt(np.logspace(np.log10(1/2000.),np.log10(kk.max()),40)[1:]*np.logspace(np.log10(1/2000.),np.log10(kk.max()),40)[:-1])
wl=1.0/kmid
fields=["10u","10v","msl"]
fig,axes=plt.subplots(1,3,figsize=(16.5,5.6),dpi=140)
for ax,fld in zip(axes,fields):
    sp0=J["b785matched"]["spectra"][fld]["spectrum_1d"]
    tr=np.array(sp0["truth"],float); inp=np.array(sp0["input"],float); n=min(len(kmid),len(tr))
    x=wl[:n]
    ax.loglog(x,tr[:n],color="#2e7d32",lw=2.8,label="truth (ENFO O1280)",zorder=6)
    ax.loglog(x,inp[:n],color="#9aa0a6",lw=1.3,ls=":",label="input (interp O320)",zorder=3)
    for a,(lab,c,p) in arms.items():
        m=np.array(J[a]["spectra"][fld]["spectrum_1d"]["model"],float)[:n]
        ax.loglog(x,m,color=c,lw=1.8,label=lab,zorder=5)
    ax.axvspan(20,100,color="#f1c40f",alpha=.13,zorder=0)
    ax.invert_xaxis(); ax.grid(which="both",alpha=.18)
    fr={a:round(J[a]["spectra"][fld]["bands"]["model"]["fine"]/J[a]["spectra"][fld]["bands"]["truth"]["fine"],2) for a in arms}
    ax.set_title(f"{fld}   fine-band ratio: b785 {fr['b785matched']} · eecdb {fr['eecdb127']} · warm {fr['warmstart']} · scr {fr['scratch512']}",fontsize=8.6)
    ax.set_xlabel("wavelength (km) — large ← → fine detail")
    if fld=="10u":
        ax.set_ylabel("radial power (per-member avg, N=50)"); ax.legend(fontsize=8,loc="lower left")
fig.suptitle("tc_o320_o1280 — full radial power spectra, ALL wavenumbers, 4 checkpoints + truth + input (box-FFT, lead 072)\n"
 "in the shaded 20–100 km fine band: b785(unified) = 3.5–4× OVER (noise-like) · eecdb127(ds-API) = ON truth · warmstart ≈ truth · scratch512 UNDER (smooth)",fontsize=10,y=1.05)
fig.text(0.5,-0.05,"eecdb127 is ALSO a global o320→o1280 model but sits on truth (fine 0.91/0.92/0.99, slope −3.27/−3.13/−4.16) → b785's fine-band excess is b785-specific, NOT inherent to global downscaling.",
         ha="center",fontsize=8,color="#333")
fig.tight_layout()
out=f"{SCR}/full_spectra_4arm.png"; fig.savefig(out,bbox_inches="tight"); print("SAVED",out)
