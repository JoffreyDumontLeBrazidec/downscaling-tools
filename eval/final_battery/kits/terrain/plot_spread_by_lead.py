import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
s=pd.read_csv('tables/T8_spread.csv'); t7=pd.read_csv('tables/T7_large_scale.csv')
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
bands=[('sea','-'),('200-500','--'),('1500-2000',':')]
names={'truth_pf':'9 km truth ENS','input':'input ENS','model':'GL512 450k'}
for a,v in zip(ax[:2],['msl','sp']):
    for b,ls in bands:
        for src,c in [('truth_pf','k'),('input','tab:blue'),('model','tab:red')]:
            d=s[(s['var']==v)&(s.zhr_band==b)&(s.source==src)].sort_values('lead')
            a.plot(d.lead,d.spread,ls,color=c,marker='o',ms=3,label=f"{names[src]} ({b}{' m' if b!='sea' else ''})")
    a.set_title(f"{v}: ensemble spread by lead (30 dates, 10 members)",fontsize=9); a.set_xlabel('lead (h)'); a.set_ylabel('Pa'); a.set_xticks([12,24,36,48,60,72])
ax[0].legend(fontsize=6)
a=ax[2]
for src,c,l in [('truth_ctrl_minus_input_ctrl','k','9 km truth control minus its input'),('model_minus_input','tab:red','GL512 draw minus its own input')]:
    d=t7[(t7.source==src)&(t7['var']=='msl')&(t7.subset=='interior')].sort_values('lead')
    a.errorbar(d.lead,d.lowpass_rms,2*d.lowpass_rms_sd_dates/30**.5,color=c,marker='o',ms=3,label=l)
a.set_title('msl: large-scale (0.6 deg low-pass) change from input',fontsize=9); a.set_xlabel('lead (h)'); a.set_ylabel('RMS (Pa), bars 2 SE over 30 dates'); a.set_xticks([12,24,36,48,60,72]); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig('figs/summary/f4_spread_by_lead.png',dpi=130)
