"""Multi-arm full radial power-spectra OVERLAY (all wavenumbers).

The exact code behind the testbed-validation spectra figure: overlays the per-arm 1-D radial
power spectrum (from each arm's <arm>_physrealism.json spectrum_1d) for 10u/10v/msl against a
shared truth + input, log-log vs wavelength, with the 20-100 km fine band shaded. Reconstructs the
identical wavenumber axis (347x747 interior, 0.075 deg, 39 log bins) so it aligns with the saved
spectra. Companion to render.py (which does the SINGLE-run per-eval version inside eval.cli).

Point ARMS/D at any set of *_physrealism.json (produced by render.py or the T24 regional_physrealism
audit). Kept here for reproducibility of the tc_o320_o1280 testbed-validation figures.
"""
import json, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="/home/ecm5702/perm/eval-rescue-20260709/regional_three_curve_verdict/physical_realism_20260710"
arms={"b785matched":"b785 (global, cropped)","warmstart":"warmstart","scratch512":"scratch512 (from-scratch)"}
col={"b785matched":"#8e44ad","warmstart":"#1a73e8","scratch512":"#c0392b","truth":"#2e7d32","input":"#9aa0a6"}
J={a:json.load(open(f"{D}/{a}_physrealism.json")) for a in arms}

# --- reconstruct the exact kmid (wavenumber axis, cycles/km) the script used ---
GRID=0.075; RIM=2.0
glat=np.arange(5+RIM,35-RIM+1e-9,GRID); glon=np.arange(-100+RIM,-40-RIM+1e-9,GRID)
ny,nx=len(glat),len(glon)
DY=GRID*111.195; DX=DY*np.cos(np.deg2rad(20.0))
ky=np.fft.fftfreq(ny,d=DY); kx=np.fft.rfftfreq(nx,d=DX)
kk=np.sqrt(ky[:,None]**2+kx[None,:]**2); kmax=kk.max()
kbins=np.logspace(np.log10(1/2000.0),np.log10(kmax),40); kmid=np.sqrt(kbins[1:]*kbins[:-1])
wl=1.0/kmid  # wavelength km
print("grid",ny,"x",nx,"kmid len",len(kmid),"wl range",wl.max(),wl.min())

fields=["10u","10v","msl"]
fig,axes=plt.subplots(1,3,figsize=(16,5.4),dpi=140)
for ax,fld in zip(axes,fields):
    sp0=J["b785matched"]["spectra"][fld]["spectrum_1d"]
    tr=np.array(sp0["truth"],float); inp=np.array(sp0["input"],float)
    n=min(len(kmid),len(tr))
    def norm(y): return np.array(y,float)[:n]
    x=wl[:n]
    ax.loglog(x,norm(tr),color=col["truth"],lw=2.6,label="truth (ENFO O1280)",zorder=5)
    ax.loglog(x,norm(inp),color=col["input"],lw=1.4,ls=":",label="input (interp O320)",zorder=3)
    for a,lab in arms.items():
        m=np.array(J[a]["spectra"][fld]["spectrum_1d"]["model"],float)[:n]
        ax.loglog(x,m,color=col[a],lw=1.8,label=lab,zorder=4)
    ax.axvspan(20,100,color="#f1c40f",alpha=.13,zorder=0)   # fine band 20-100 km
    ax.axvline(1000,color="#888",ls="--",lw=.7,alpha=.6); ax.axvline(100,color="#888",ls="--",lw=.7,alpha=.6)
    ax.text(46,ax.get_ylim()[1]*0.4 if False else 0,"",fontsize=7)
    ax.set_title(f"{fld}",fontsize=12)
    ax.set_xlabel("wavelength (km)  —  large scales ← → fine detail")
    ax.invert_xaxis()            # so fine detail is on the RIGHT
    ax.grid(which="both",alpha=.18)
    if fld=="10u":
        ax.set_ylabel("radial power (per-member avg, N=50)")
        ax.legend(fontsize=8.2,loc="lower left")
    # band labels
    ax.annotate("fine\n20–100 km",(46,ax.get_ylim()[0]*3),fontsize=7.5,color="#b7950b",ha="center")
fig.suptitle("tc_o320_o1280 — full radial power spectra, ALL wavenumbers (T24 box-FFT, lead 072 h)\n"
 "truth over ALL curves; in the shaded 20–100 km fine band: b785 sits ABOVE truth (excess), warmstart tracks truth, scratch512 falls BELOW (over-smooth), input collapses",
 fontsize=10.5,y=1.06)
method=("METHOD: 284,713 native O1280 box pts → nearest-neighbour onto a regular 0.075° grid "
 f"({ny}×{nx} interior, 2° rim removed, max NN 6.3 km) · per field: linear-plane detrend + 2D Hann window · "
 "np.fft.rfft2 power · isotropic radial binning into 39 log-spaced wavenumber bins (kmin=1/2000, kmax≈0.088 cyc/km) · "
 "per-MEMBER spectra averaged (never the ensemble-mean field) · N = 5 dates × 10 members = 50 · truth = each member's ENFO-O1280 target y.")
fig.text(0.5,-0.06,method,ha="center",va="top",fontsize=7.6,color="#333",wrap=True)
fig.tight_layout()
out=f"{os.path.expanduser('~')}/scratch/eval/testbed_validation_plots/full_spectra_allwn.png"
fig.savefig(out,bbox_inches="tight"); print("SAVED",out)
