# Projective Quantum Realization in TCGS–SEQUENTION

**Anomalies, Global Coherence, and Invariant Readout**  
Henry Arellano-Peña · Nuevo Estándar Biotropical NEBIOT S.A.S., Bogotá, Colombia  
Version 4 · 12 November 2026

[Read or download the 41-page manuscript](TCGS_SEQUENTION_Projective_Quantum_Realization.pdf) · [References](REFERENCES.md) · [License](LICENSE.txt)

The paper studies when a globally smooth profile of quantum density operators admits a globally continuous normalized amplitude. In a specified qubit realization, the first Chern number is −1 for |a| < 1 and 0 for |a| > 1, where a is a scalar obtained from the source by a calibrated readout. The normalized construction is singular at a = ±1. The density profile already determines a, and the Chern number classifies the existence of a global amplitude lift.

The manuscript also gives sufficient conditions for Born weights through probability transfer across calibrated POVMs, including for qubits, and illustrates unique but unstable reconstruction from harmonic boundary data. These results apply under the stated realization and measurement hypotheses.

**Suggested citation.** Henry Arellano-Peña (2026), *Projective Quantum Realization in TCGS–SEQUENTION: Anomalies, Global Coherence, and Invariant Readout*, version 4, manuscript dated 12 November 2026.

## Reproduce the manuscript checks

The [Python verification script](verify_examples.py) runs the 15 symbolic check groups for version 4. These include the source-dependent Chern number, the qubit POVM counterexample, and the harmonic reconstruction example.

From the repository root, install the pinned dependency and run the script:

```bash
python -m pip install -r requirements.txt
python verify_examples.py --json
```

[Recorded verification results](verification_results.json)
