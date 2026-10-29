#!/usr/bin/env python3
"""EFMW-RP-001. Reproduce the frozen synthetic test; no observed data are used."""
from pathlib import Path
import argparse, base64, csv, hashlib, html, json, platform, time
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
EXPECTED = (ROOT / 'PROTOCOL_SHA256.txt').read_text().split()[0]
assert hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest() == EXPECTED
P = PROTOCOL['parameters']
rho, R0, p0, pv = [P[x] for x in ['rho_kg_m3', 'R0_m', 'P0_Pa', 'pv_Pa']]
sigma = P['surface_tension_N_m']
pg = p0-pv+2*sigma/R0
m = rho*R0**2
k = 3*P['kappa']*pg-2*sigma/R0
b = 4*P['eta_effective_Pa_s']
dk = 4*P['G_memory_Pa']
w0 = np.sqrt(k/m)
B = b*w0/k
FREQ = np.geomspace(2000., 200000., 61)
OMEGA = 2*np.pi*FREQ
TRAIN = np.arange(61) % 2 == 0
TEST = ~TRAIN
NOISE_H = PROTOCOL['design']['sigma_H_per_quadrature_Pa_inverse']
NOISE = NOISE_H*k
TRUE0 = np.array([1., B])
TRUE1 = np.array([1., B, dk/k, np.log10(P['tau_s'])])


def model(p, omega):
    """Dimensionless k*H, for exp(+i omega t) phasors."""
    u = omega/w0
    den = p[0]-u*u+1j*p[1]*u
    if len(p) == 4:
        z = 1j*omega*10**p[3]
        den = den+p[2]*z/(1+z)
    return -1/den


def resid(p, omega, y):
    z = (model(p, omega)-y)/NOISE
    return np.r_[z.real, z.imag]


def jac(p, omega, y):
    u = omega/w0
    h = model(p, omega)
    cols = [np.ones_like(u, dtype=complex), 1j*u]
    if len(p) == 4:
        z = 1j*omega*10**p[3]
        cols += [z/(1+z), p[2]*np.log(10)*z/(1+z)**2]
    j = np.stack(cols, axis=1)*h[:, None]**2/NOISE
    return np.r_[j.real, j.imag]


def fit_pair(y):
    om, yt = OMEGA[TRAIN], y[TRAIN]
    r0 = least_squares(resid, TRUE0, jac=jac, args=(om, yt),
                       bounds=([.5, 1e-5], [2., 1.5]),
                       ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=400)
    if not r0.success:
        raise RuntimeError('Baseline optimization failed: '+r0.message)
    cost0 = float(np.sum(r0.fun**2))
    # Explicitly retain the nested model so a poor optimizer cannot worsen M1.
    p1 = np.r_[r0.x, 0., -5.]
    cost1 = cost0
    failed_starts = 0
    for logtau in (-6., -5., -4.):
        start = [max(.501, r0.x[0]-.05), r0.x[1], .05, logtau]
        r = least_squares(resid, start, jac=jac, args=(om, yt),
                          bounds=([.5, 1e-5, 0., -7.], [2., 1.5, 1., -3.]),
                          ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=400)
        if not r.success:
            failed_starts += 1
            continue
        cost = float(np.sum(r.fun**2))
        if cost < cost1:
            cost1, p1 = cost, r.x.copy()
    e0 = np.sqrt(np.mean(resid(r0.x, OMEGA[TEST], y[TEST])**2))
    e1 = np.sqrt(np.mean(resid(p1, OMEGA[TEST], y[TEST])**2))
    return dict(p0=r0.x, p1=p1, delta_chi2=cost0-cost1,
                wrmse0=float(e0), wrmse1=float(e1), improvement=float(1-e1/e0),
                failed_starts=failed_starts)


def noisy(rng, truth):
    return model(truth, OMEGA)+NOISE*(rng.normal(size=61)+1j*rng.normal(size=61))


def row(stage, replicate, fit, cutoff=None):
    detected = None if cutoff is None else bool(
        fit['delta_chi2'] > cutoff and fit['improvement'] >= .2 and fit['wrmse1'] <= 1.5)
    p = fit['p1']
    return dict(stage=stage, replicate=replicate,
                delta_chi2=fit['delta_chi2'], heldout_wrmse_baseline=fit['wrmse0'],
                heldout_wrmse_memory=fit['wrmse1'], heldout_rmse_reduction=fit['improvement'],
                relaxation_detected=detected, fitted_tau_us=10**p[3]*1e6,
                fitted_G_Pa=p[2]*k/4, fitted_k_Pa=p[0]*k,
                fitted_b_Pa_s=p[1]*k/w0, failed_optimizer_starts=fit['failed_starts'])


def wilson(hits, n):
    z=1.959963984540054
    phat=hits/n
    center=(phat+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half), min(1.,center+half)]


def summarize(rows):
    hits=sum(r['relaxation_detected'] for r in rows)
    n=len(rows)
    def quant(key):
        return dict(zip(['p2_5','median','p97_5'], map(float, np.quantile([r[key] for r in rows], [.025,.5,.975]))))
    return dict(n=n, detected=int(hits), detection_fraction=hits/n,
                detection_fraction_wilson95=wilson(hits,n),
                heldout_rmse_reduction=quant('heldout_rmse_reduction'),
                heldout_wrmse_baseline=quant('heldout_wrmse_baseline'),
                heldout_wrmse_memory=quant('heldout_wrmse_memory'),
                fitted_tau_us=quant('fitted_tau_us'), fitted_G_Pa=quant('fitted_G_Pa'),
                failed_optimizer_starts=sum(r['failed_optimizer_starts'] for r in rows))


def ode_response(freq, memory=False, nonlinear=False):
    """Independent time integration; project final cycles onto sin/cos."""
    u = 2*np.pi*freq/w0
    theta = P['tau_s']*w0
    g = dk/k if memory else 0.
    A = P['drive_amplitude_Pa']
    matrix = np.array([[0,1,0],[-1,-B,-1],[0,g,-1/theta]])
    decay = float(-np.max(np.linalg.eigvals(matrix).real))
    period = 2*np.pi/u
    start = np.ceil(max(20*period, 25/decay)/period)*period
    end = start+12*period
    samples = np.linspace(start,end,2400,endpoint=False)
    if nonlinear:
        def fun(s,y):
            r,v=y
            if r <= 0:
                raise RuntimeError('Nonphysical radius in validation')
            pressure=pg*r**(-3*P['kappa'])+pv-p0-A*np.cos(u*s)-2*sigma/(R0*r)-b*w0*v/r
            return [v,pressure/(k*r)-1.5*v*v/r]
        initial=[1.,0.]
    else:
        def fun(s,y):
            x,v,z=y
            return [v,-B*v-x-z-A/k*np.cos(u*s),g*v-z/theta]
        initial=[0.,0.,0.]
    sol=solve_ivp(fun,[0,end],initial,method='DOP853',t_eval=samples,
                  rtol=1e-10,atol=1e-12,max_step=period/32)
    if not sol.success:
        raise RuntimeError(sol.message)
    x=sol.y[0]-(1 if nonlinear else 0)
    design=np.c_[np.cos(u*samples), np.sin(u*samples), np.ones_like(samples)]
    coef=np.linalg.lstsq(design,x,rcond=None)[0]
    numerical=(coef[0]-1j*coef[1])/A
    analytic=model(TRUE1 if memory else TRUE0,np.array([2*np.pi*freq]))[0]/k
    return dict(freq_Hz=float(freq), model='nonlinear_RP' if nonlinear else ('linear_memory' if memory else 'linear_null'),
                numerical_real=float(numerical.real), numerical_imag=float(numerical.imag),
                analytic_real=float(analytic.real), analytic_imag=float(analytic.imag),
                relative_complex_error=float(abs(numerical-analytic)/abs(analytic)),
                max_fractional_radius_sampled=float(np.max(abs(x))), nfev=int(sol.nfev))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def make_figure(out, result, rows, example):
    navy, teal, rust, gray = '#24364b','#008675','#bd583b','#6b7280'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':navy,'text.color':navy})
    fig,axes=plt.subplots(2,2,figsize=(13,9.4))
    grid=np.geomspace(2000.,200000.,1500)
    h0=model(example['fit']['p0'],2*np.pi*grid)/k
    h1=model(example['fit']['p1'],2*np.pi*grid)/k
    hx=model(TRUE1,2*np.pi*grid)/k
    obs=example['y']/k
    ax=axes[0,0]
    ax.loglog(grid/1000,abs(h0)*1e6,color=rust,ls='--',label='Fitted constant-damping baseline')
    ax.loglog(grid/1000,abs(h1)*1e6,color=teal,label='Fitted exponential memory')
    ax.scatter(FREQ[TEST]/1000,abs(obs[TEST])*1e6,s=17,facecolor='white',edgecolor=navy,label='Synthetic held-out observations',zorder=3)
    ax.set(xlabel='Driving frequency (kHz)',ylabel='|fractional radius / pressure| (10⁻⁶ Pa⁻¹)',title='A  |  Prediction at held-out frequencies')
    ax.legend(fontsize=8,loc='lower left')
    ax=axes[0,1]
    ax.semilogx(grid/1000,np.angle(h0,deg=True),color=rust,ls='--',label='Baseline')
    ax.semilogx(grid/1000,np.angle(h1,deg=True),color=teal,label='Memory fit')
    ax.semilogx(grid/1000,np.angle(hx,deg=True),color=navy,ls=':',label='Conventional Maxwell generating model')
    ax.scatter(FREQ[TEST]/1000,np.angle(obs[TEST],deg=True),s=17,facecolor='white',edgecolor=navy)
    ax.axvline(1/(2*np.pi*P['tau_s'])/1000,color=gray,lw=.8,ls=':')
    ax.set(xlabel='Driving frequency (kHz)',ylabel='Radius phase relative to pressure (degrees)',title='B  |  The same phase response has a classical explanation',ylim=(-8,188))
    ax.legend(fontsize=8,loc='lower left')
    null=[r['heldout_rmse_reduction']*100 for r in rows if r['stage']=='null_test']
    pos=[r['heldout_rmse_reduction']*100 for r in rows if r['stage']=='memory_test']
    ax=axes[1,0]
    bins=np.linspace(min(-25,min(null)-2,min(pos)-2),100,45)
    ax.hist(null,bins=bins,alpha=.75,color=gray,label='250 independent no-memory trials')
    ax.hist(pos,bins=bins,alpha=.8,color=teal,label='250 injected-memory trials')
    ax.axvline(20,color=rust,ls='--',label='20% prediction-gain criterion')
    ax.set(xlabel='Held-out RMSE reduction (%)',ylabel='Synthetic trials',title='C  |  Detection checks under the frozen noise model')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    taus=[r['fitted_tau_us'] for r in rows if r['stage']=='memory_test']
    ax.hist(taus,bins=23,color=teal,alpha=.8)
    ax.axvline(8,color=navy,ls='--',label='Injected relaxation time: 8 μs')
    ax.set(xlabel='Fitted relaxation time (μs)',ylabel='Synthetic trials',title='D  |  Parameter recovery; no physical measurement')
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.15)
    fig.suptitle('EFMW-RP-001  •  Synthetic bubble-response test',fontsize=19,fontweight='bold',y=.985)
    fig.text(.5,.942,'Detecting relaxation does not identify EFMW: the chosen memory law is exactly Maxwell viscoelasticity.',ha='center',fontsize=11)
    fig.text(.5,.018,'All observations are simulated. Fixed 61-frequency design: 31 fit + 30 held out. First injected-memory trial shown, without selection.',ha='center',fontsize=9,color=gray)
    fig.tight_layout(rect=[0,.04,1,.925],h_pad=2.8,w_pad=2)
    fig.savefig(out/'EFMW-RP-001-results.png',dpi=180)
    plt.close(fig)


def make_report(out, result):
    n=result['null_test']; s=result['memory_test']; v=result['verification']
    med=s['heldout_rmse_reduction']['median']*100
    tau=s['fitted_tau_us']
    image=base64.b64encode((out/'EFMW-RP-001-results.png').read_bytes()).decode()
    rows=[('No injected memory',f"{n['detected']}/{n['n']}",f"{n['heldout_rmse_reduction']['median']*100:.2f}%"),
          ('Injected Maxwell memory',f"{s['detected']}/{s['n']}",f"{med:.2f}%")]
    trs=''.join('<tr>'+''.join(f'<td>{html.escape(t)}</td>' for t in r)+'</tr>' for r in rows)
    lim=''.join(f'<li>{html.escape(x)}</li>' for x in PROTOCOL['limitations'])
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EFMW-RP-001 — computational test</title>
<style>body{{max-width:1050px;margin:36px auto;padding:0 24px;color:#24364b;font:17px/1.55 system-ui,sans-serif}}h1{{line-height:1.15}}h2{{margin-top:2em}}.lead{{border-left:5px solid #008675;padding:12px 20px;background:#f0f8f6}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border-bottom:1px solid #d7dde3;padding:10px;text-align:left}}code,pre{{font-size:14px}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:18px;border-radius:8px}}img{{width:100%}}small{{color:#64748b}}a{{color:#007a70}}@media print{{body{{font-size:11pt}}h2{{break-after:avoid}}img{{break-inside:avoid}}}}</style>
<h1>EFMW-RP-001<br>Can a bubble-response test identify the proposed memory?</h1>
<p><strong>Executed October 29, 2026.</strong> Computational feasibility and structural-identifiability study.</p>
<div class="lead"><strong>Result:</strong> the frozen analysis detects the injected relaxation at this parameter/noise setting. The selected exponential-memory law is exactly conventional Maxwell viscoelasticity. Its detection cannot independently identify EFMW, TUFT, or a universal pulsation. No laboratory experiment was performed.</div>
<h2>What was run</h2><p>500 synthetic null simulations calibrated a training-fit threshold. A separate 250 null trials and 250 injected-memory trials tested the frozen decision rule. Each trial used 61 frequencies from 2 to 200 kHz: 31 interleaved frequencies for fitting and 30 for held-out interpolation. Independent Gaussian noise of 5×10⁻⁸ Pa⁻¹ was added to each real and imaginary response component. These are analyst-selected conditions, not a proposed instrument's measured noise floor.</p>
<table><tr><th>Synthetic condition</th><th>Relaxation detections</th><th>Median held-out RMSE reduction</th></tr>{trs}</table>
<p>Null detection fraction 95% Wilson interval: {n['detection_fraction_wilson95'][0]:.4f}–{n['detection_fraction_wilson95'][1]:.4f}. Injected-memory detection fraction interval: {s['detection_fraction_wilson95'][0]:.4f}–{s['detection_fraction_wilson95'][1]:.4f}. Recovery of the injected 8 μs timescale: median {tau['median']:.3f} μs; central 95% of fitted values {tau['p2_5']:.3f}–{tau['p97_5']:.3f} μs. This spread is a Monte Carlo recovery distribution, not a confidence interval for a physical measurement.</p>
<img alt="Four-panel plot of synthetic frequency response, phase, held-out error improvements, and fitted relaxation times" src="data:image/png;base64,{image}">
<h2>Model and dimensional derivation</h2><p>Linearize the Rayleigh–Plesset equation about an existing equilibrium bubble with x=(R−R₀)/R₀. The gas/vapor closure is pB=pv+pg0(R₀/R)^(3κ). Equilibrium requires pg0=P₀−pv+2σ/R₀. To first order the ordinary equation is m ẍ+b ẋ+k x=−pdrive, with m=ρR₀², b=4ηeff, k=3κpg0−2σ/R₀. Thus m has units Pa·s², b Pa·s, and k Pa.</p>
<pre>Memory extension (specified for this test; not derived from EFMW):
m ẍ + b ẋ + k x + q = −pdrive(t)
τ q̇ + q = Δk τ ẋ,  where Δk = 4 Gm

Steady-state response, exp(+iωt) convention:
H(ω) = x̂/p̂ = −1 / [k − mω² + i bω + Δk iωτ/(1+iωτ)]

Exponential history form, after initial transients:
q(t) = Δk ∫₀^∞ exp(−s/τ) ẋ(t−s) ds

For the small-amplitude passive model:
E = ½m ẋ² + ½k x² + q²/(2Δk)
dE/dt = −b ẋ² − q²/(Δk τ) − pdrive ẋ</pre>
<p>The energy identity demonstrates passive relaxation when m,k,b,Δk,τ are positive. Driving energy enters through the prescribed pressure; this equation does not derive a self-sustaining global pulsation.</p>
<h2>The decisive identifiability failure</h2><p>A conventional Maxwell element has G*(ω)=Gm iωτ/(1+iωτ). The linear spherical-bubble contribution is 4G*(ω), giving exactly the same H(ω). This is algebraic equivalence, independent of noise, frequency selection, or fitting. Comparing differently named copies of that equation cannot distinguish their interpretation. Numerical relative mismatch on 10,001 frequencies: {v['maxwell_max_relative_difference']:.3g}.</p>
<p>A second illustrative counterexample puts the same pole into a lumped gas-relaxation exponent κ*(ω)=κ+[Δk/(3pg0)]iωτ/(1+iωτ). Here κ runs from 1.2 to {v['thermal_surrogate_kappa_high']:.6f}, and the response also matches algebraically. This surrogate illustrates a confounder; it is not a replacement for solving gas heat diffusion. Classical bubble theory already includes frequency-dependent thermal stiffness and damping.</p>
<h2>Frozen success/failure rule</h2><p>A trial counted as detecting relaxation only if (1) its training Δχ² exceeded the independent-null 95th-percentile threshold ({result['calibration_delta_chi2_cutoff']:.4f}), (2) held-out weighted RMSE improved by at least 20%, and (3) memory-model held-out weighted RMSE was at most 1.5. The baseline fit both stiffness and damping; the memory model additionally fit Δk and τ. The held-out data were never used for fitting or choosing parameters. Calibration was simulated because τ is unidentified when Δk=0, invalidating a naive regular likelihood-ratio reference distribution.</p>
<p>Algorithm check: <strong>{result['algorithm_check']}</strong> under the prespecified ≤5% null and ≥90% positive-detection criteria. Uniqueness check: <strong>FAIL</strong> for the selected model, because of the exact Maxwell counterpart. Physical-truth check: <strong>UNRESOLVED</strong>.</p>
<h2>Numerical verification</h2><p>Independent time-domain integration of the linear differential equations was compared with the frequency-domain transfer at 5 kHz, 20 kHz, nominal resonance, and 150 kHz, for both models. Maximum relative complex discrepancy: {v['ode_max_relative_error']:.3g}; criterion 10⁻⁵. The nonlinear Newtonian Rayleigh–Plesset equation was independently integrated at the same frequencies with a 20 Pa drive. Its first harmonic differed from the linear baseline by at most {v['nonlinear_RP_max_relative_error']:.3g}; criterion 0.01. Maximum predicted fractional radius amplitude over the dense frequency grid: {v['maximum_fractional_radius_amplitude']:.5f}; criterion 0.01. Numerical checks: <strong>{v['overall_status']}</strong>.</p>
<h2>Interpretation and next experiment</h2><p>This computation demonstrates a detector's behavior when a chosen conventional relaxation term is inserted into synthetic data. It does not establish that such a term exists in nature, and a synthetic positive control cannot supply independent support for its own generating assumptions.</p>
<p>For an empirical test, independently characterize bubble radius, pressure calibration, fluid complex rheology, gas thermal response, surface tension, and measurement noise. Specify the full conventional baseline and a quantitatively different EFMW term before collecting the blinded held-out frequency sweep. A claimed anomaly must survive those controls and independent replication. If EFMW supplies no response different from the conventional model, this observable cannot identify it regardless of measurement precision.</p>
<h2>Audit operations</h2><table><tr><th>Operation</th><th>Input / criterion</th><th>Output / evidence status</th></tr>
<tr><td>Dimensions and passivity</td><td>ODE coefficients; consistent pressure units and nonnegative dissipation</td><td>Demonstrated for the chosen model</td></tr><tr><td>Numerical cross-check</td><td>Time-domain ODE versus transfer and nonlinear RP first harmonic; frozen tolerances</td><td>{v['overall_status']}; computation</td></tr><tr><td>Null calibration and held-out prediction</td><td>Independent seeds, frozen split and thresholds</td><td>{result['algorithm_check']}; synthetic only</td></tr><tr><td>Conventional-model equivalence</td><td>Maxwell constitutive law; distinguishable transfer required</td><td>Uniqueness rejected for this response law</td></tr><tr><td>Physical identification</td><td>Independent measurements and EFMW-derived unique prediction required</td><td>Unresolved; neither supplied</td></tr></table>
<h2>Limitations retained</h2><ul>{lim}</ul>
<h2>Sources and novelty check</h2><p>Known overlap is equivalent/reparameterized conventional linear viscoelasticity. No novelty is claimed for the response law, fit method, or simulated signal.</p><ul>
<li>Allen &amp; Roy (2000), <a href="https://doi.org/10.1121/1.429344">Dynamics of gas bubbles in viscoelastic fluids. I. Linear viscoelasticity</a>. Maxwell/Jeffreys bubble models and phase response predate this proposal.</li>
<li>Warnez &amp; Johnsen (2015), <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4474959/">Numerical modeling of bubble dynamics in viscoelastic media with relaxation</a>. Relevant conventional model family; not an assertion that our simplified ODE reproduces their full finite-amplitude model.</li>
<li>Brennen, <a href="https://media.library.caltech.edu/CaltechBOOK:1995.001/chap2.htm">Cavitation and Bubble Dynamics, Chapter 2</a>, and <a href="https://media.library.caltech.edu/CaltechBOOK:1995.001/chap4.htm">Chapter 4</a>. Rayleigh–Plesset baseline, linear oscillations, thermal effects and damping.</li></ul>
<h2>Reproducibility</h2><p>Run <code>python run_test.py</code> with the versions in requirements.txt. Full protocol, seeds, bounds, all 1,000 trial rows, 500 synthetic test spectra, representative predictions, integration checks, and machine-readable results are included. The local pre-execution protocol hash is <code>{EXPECTED}</code>. This is an internal freeze, not an externally timestamped preregistration.</p>
<small>Generated by a reproducible computational workflow for Matthew Wright. No laboratory or independent empirical validation is claimed.</small></html>'''
    (out/'report.html').write_text(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    rows=[]
    rng=np.random.default_rng(PROTOCOL['design']['seeds']['calibration'])
    for i in range(PROTOCOL['design']['null_calibration_replicates']):
        fit=fit_pair(noisy(rng,TRUE0))
        rows.append(row('calibration',i,fit))
        if (i+1)%100==0:
            print(f'Calibration {i+1}/500, elapsed {time.time()-started:.1f}s',flush=True)
    cutoff=float(np.quantile([r['delta_chi2'] for r in rows],.95,method='higher'))
    print(f'Frozen calibration threshold: {cutoff:.6f}',flush=True)
    spectra=[]; example=None
    for stage,truth,n in [('null_test',TRUE0,250),('memory_test',TRUE1,250)]:
        rng=np.random.default_rng(PROTOCOL['design']['seeds'][stage])
        for i in range(n):
            y=noisy(rng,truth)
            fit=fit_pair(y)
            rows.append(row(stage,i,fit,cutoff))
            for j,f in enumerate(FREQ):
                spectra.append(dict(stage=stage,replicate=i,freq_Hz=float(f),split='train' if TRAIN[j] else 'held_out',
                                    H_real=float(y[j].real/k),H_imag=float(y[j].imag/k),sigma_H=NOISE_H))
            if stage=='memory_test' and i==0:
                example=dict(y=y,fit=fit)
            if (i+1)%50==0:
                print(f'{stage} {i+1}/{n}, elapsed {time.time()-started:.1f}s',flush=True)
    null=summarize([r for r in rows if r['stage']=='null_test'])
    pos=summarize([r for r in rows if r['stage']=='memory_test'])
    print('Running independent ODE/linearization checks',flush=True)
    checks=[]
    for f in [5000.,20000.,w0/(2*np.pi),150000.]:
        checks += [ode_response(f),ode_response(f,memory=True),ode_response(f,nonlinear=True)]
    om=2*np.pi*np.geomspace(2000.,200000.,10001)
    z=1j*om*P['tau_s']
    hm=model(TRUE1,om)/k
    gstar=P['G_memory_Pa']*z/(1+z)
    hmaxwell=-1/(k-m*om**2+1j*b*om+4*gstar)
    kappa_star=P['kappa']+(dk/(3*pg))*z/(1+z)
    hthermal=-1/(3*pg*kappa_star-2*sigma/R0-m*om**2+1j*b*om)
    equivalence=float(np.max(abs(hm-hmaxwell)/abs(hm)))
    thermal_diff=float(np.max(abs(hm-hthermal)/abs(hm)))
    ode_err=max(c['relative_complex_error'] for c in checks if c['model']!='nonlinear_RP')
    rp_err=max(c['relative_complex_error'] for c in checks if c['model']=='nonlinear_RP')
    maxamp=float(P['drive_amplitude_Pa']*max(np.max(abs(hm)),np.max(abs(model(TRUE0,om)/k))))
    matrix=np.array([[0,1,0],[-1,-B,-1],[0,dk/k,-1/(P['tau_s']*w0)]])
    eig=np.linalg.eigvals(matrix)*w0
    ver=dict(maxwell_max_relative_difference=equivalence,
             thermal_surrogate_max_relative_difference=thermal_diff,
             thermal_surrogate_kappa_high=P['kappa']+dk/(3*pg),
             ode_max_relative_error=ode_err,nonlinear_RP_max_relative_error=rp_err,
             maximum_fractional_radius_amplitude=maxamp,
             memory_eigenvalues_per_s=[dict(real=float(e.real),imag=float(e.imag)) for e in eig],
             overall_status='PASS' if ode_err<1e-5 and rp_err<.01 and maxamp<.01 and equivalence<1e-12 and thermal_diff<1e-12 else 'FAIL')
    result=dict(study_id=PROTOCOL['study_id'],data_status='SYNTHETIC_ONLY',
                protocol_sha256=EXPECTED,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
                parameters=P,derived=dict(m_Pa_s2=m,k_Pa=k,b_Pa_s=b,delta_k_Pa=dk,pg0_Pa=pg,
                                         nominal_resonance_Hz=w0/(2*np.pi),relaxation_corner_Hz=1/(2*np.pi*P['tau_s'])),
                calibration_n=500,calibration_delta_chi2_cutoff=cutoff,null_test=null,memory_test=pos,
                algorithm_check='PASS' if null['detection_fraction']<=.05 and pos['detection_fraction']>=.9 else 'FAIL',
                uniqueness_status='FAIL_EXACT_CONVENTIONAL_MAXWELL_EQUIVALENCE',
                physical_truth_status='UNRESOLVED_NO_EMPIRICAL_DATA',verification=ver,
                all_failed_optimizer_starts=sum(r['failed_optimizer_starts'] for r in rows),
                elapsed_seconds=time.time()-started)
    write_csv(out/'trials.csv',rows)
    write_csv(out/'synthetic_spectra.csv',spectra)
    write_csv(out/'ode_verification.csv',checks)
    example_rows=[]
    for j,f in enumerate(FREQ):
        preds=[model(example['fit'][p],np.array([OMEGA[j]]))[0]/k for p in ['p0','p1']]
        example_rows.append(dict(freq_Hz=float(f),split='train' if TRAIN[j] else 'held_out',
                                 synthetic_H_real=float(example['y'][j].real/k),synthetic_H_imag=float(example['y'][j].imag/k),
                                 baseline_H_real=float(preds[0].real),baseline_H_imag=float(preds[0].imag),
                                 memory_H_real=float(preds[1].real),memory_H_imag=float(preds[1].imag)))
    write_csv(out/'representative_predictions.csv',example_rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    make_figure(out,result,rows,example)
    make_report(out,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
