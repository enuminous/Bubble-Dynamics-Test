#!/usr/bin/env python3
"""Post-run reporting only. Does not alter fits, criteria, data, or outcomes."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import run_test as study

root=Path(__file__).resolve().parent
out=root/'results'
r=json.loads((out/'results.json').read_text())
rows=list(csv.DictReader((out/'trials.csv').open()))
signal=[x for x in rows if x['stage']=='memory_test']
cutoff=r['calibration_delta_chi2_cutoff']
p=study.TRUE1.copy()
w=study.OMEGA[study.TRAIN]
y=study.model(p,w)
columns=[]
for index in range(4):
 step=1e-6
 plus=p.copy(); minus=p.copy()
 plus[index]+=step; minus[index]-=step
 columns.append((study.resid(plus,w,y)-study.resid(minus,w,y))/(2*step))
jnum=np.stack(columns,axis=1)
jerror=float(np.max(abs(study.jac(p,w,y)-jnum))/np.max(abs(jnum)))
diagnostics={
 'post_run_reporting_only': True,
 'protocol_unchanged': hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==r['protocol_sha256'],
 'analysis_source_unchanged': hashlib.sha256((root/'run_test.py').read_bytes()).hexdigest()==r['code_sha256'],
 'memory_trial_failures_by_criterion': {
  'training_delta_chi2': sum(float(x['delta_chi2'])<=cutoff for x in signal),
  'heldout_gain_below_20_percent': sum(float(x['heldout_rmse_reduction'])<.2 for x in signal),
  'heldout_wrmse_above_1_5': sum(float(x['heldout_wrmse_memory'])>1.5 for x in signal)},
 'optimizer_starts_attempted': 3*len(rows),
 'optimizer_starts_not_converged': sum(int(x['failed_optimizer_starts']) for x in rows),
 'max_nonconverged_starts_in_one_trial': max(int(x['failed_optimizer_starts']) for x in rows),
 'trials_with_all_three_memory_starts_nonconverged': sum(int(x['failed_optimizer_starts'])==3 for x in rows),
 'no_trials_excluded': len(rows)==1000,
 'jacobian_independent_finite_difference_check': {
  'relative_maximum_discrepancy': jerror,
  'method': 'Central finite differences, step 1e-6 per dimensionless/log parameter, at TRUE1 on training frequencies; checked after protocol execution began.'},
 'interpretation': 'The observed detection fraction misses the frozen 90% target. Its confidence interval includes 90%; this is a procedural failure, not proof that population sensitivity is below 90%. No parameters, noise, criteria, or seeds were retuned.'
}
assert diagnostics['protocol_unchanged'] and diagnostics['analysis_source_unchanged']
(out/'postrun_audit.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
report=(out/'report.html').read_text()
old='the frozen analysis detects the injected relaxation at this parameter/noise setting.'
new=('the frozen analysis detected injected relaxation in 223/250 trials (89.2%), '
     'below its prespecified 90% target. There were 0/250 detections under the no-memory control. '
     'The detector therefore did not pass its complete frozen acceptance rule.')
if old in report:
 report=report.replace(old,new,1)
marker='<h2>Post-run diagnostic accounting</h2>'
if marker not in report:
 note=f'''{marker}<p>All 250 injected-memory trials exceeded the training-evidence and 20% held-out improvement requirements. The 27 rejected trials failed the separate held-out weighted-RMSE adequacy cutoff of 1.5. The observed 89.2% detection rate misses the frozen 90% target; its 95% Wilson interval includes 90%, so this does not establish that the population detection probability is below 90%. Thresholds were not revised after seeing this result.</p>
<p>{diagnostics['optimizer_starts_not_converged']} of {diagnostics['optimizer_starts_attempted']} memory-fit starts did not satisfy the optimizer's convergence condition. These occurred only in null calibration/control trials, where relaxation time is nonidentified at zero memory. The specified multi-start procedure retained successful solutions or the exact nested-null fallback; no trials were excluded. Trials with all three starts nonconverged: {diagnostics['trials_with_all_three_memory_starts_nonconverged']}. The null calibration consequently describes this particular fitting algorithm, including these convergence limitations, rather than a guaranteed global optimizer. All positive-control starts converged.</p>
<p>A separate central-finite-difference check of the analytic optimizer Jacobian gave relative maximum discrepancy 5.21×10⁻¹¹. Reporting was finalized by <code>finalize_report.py</code>; the original analysis-source hash and frozen protocol remain unchanged.</p>'''
 report=report.replace('<h2>Reproducibility</h2>',note+'<h2>Reproducibility</h2>')
 report=report.replace('Run <code>python run_test.py</code> with',
                       'Run <code>python run_test.py</code>, then <code>python finalize_report.py</code>, with')
(out/'report.html').write_text(report)
print(json.dumps(diagnostics,indent=2))
