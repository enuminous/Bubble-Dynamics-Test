EFMW-RP-001 — synthetic bubble-response discrimination test
Date: October 29, 2026

Start with results/report.html (self-contained; open in a browser).
All observations are synthetic. No laboratory experiment was conducted.
The exponential memory kernel was specified by the analyst, not derived
from EFMW or TUFT. It is equivalent to classical Maxwell viscoelasticity.

REPRODUCE
Python 3.12 was used.
Install the pinned scientific dependencies in requirements.txt.
Run: python run_test.py
Then: python finalize_report.py
Optional separate output: python run_test.py --output replication
The reporting finalizer targets the default results directory; a separate
replication directory contains the unamended machine-generated report.

CONTENTS
protocol.json: hypotheses, assumptions, frequency split, noise, seeds,
parameter bounds, decision rules, limitations; frozen before execution.
PROTOCOL_SHA256.txt: internal pre-execution protocol fingerprint.
run_test.py: synthetic generator, training-only fits, independent Monte
Carlo controls, independent time integration and RP linearization checks.
results/trials.csv: all calibration and independent test-trial results.
results/synthetic_spectra.csv: all 500 independent test spectra, labeled
as simulated, with frequency splits and noise scale.
results/representative_predictions.csv: first injected-memory trial,
selected by replicate index before results were known.
results/ode_verification.csv: time-domain numerical validation.
results/results.json: complete machine-readable summary and source hash.
results/postrun_audit.json: criterion-failure and optimizer accounting.
finalize_report.py: reporting-only clarification; no refitting or retuning.
results/EFMW-RP-001-results.png: figure, explicitly labeled synthetic.
results/report.html: explanation, derivation, limitations and sources.

INTERPRETATION
Passing a synthetic detection check demonstrates the behavior of the
chosen analysis under its generating assumptions. It does not establish
a physical phenomenon. Exact equality to a conventional constitutive
model means that this observable alone cannot identify an EFMW mechanism.
The proposed Stoney-scale global driver, pressure-regime switch, and
identification of cavitation with matter were not defined sufficiently
to simulate or test here and remain unresolved.
