# ECMPP-Mine-Planning

Environmentally Constrained Mine Planning Problem (ECMPP) - NP-completeness proof and SAT/SMT implementation.

## Overview

This repository contains code and data accompanying the paper:

> "NP-Completeness of Mine Planning Under Logical Environmental Constraints"  
> Kudzawu-D'Pherdd et al., *Computers & Geosciences* (under review)

## Contents

- `ecmpp_encoder.py` - Z3 SMT encoding of the ECMPP decision problem
- `instance_generator.py` - Scalable ECMPP instance generator
- `scaling_experiment.py` - Scaling experiment runner
- `25block_example/` - The pedagogical 25-block worked example (Appendix A)
  - `block_model.csv` - 25-block economic and environmental parameters
  - `clauses.txt` - Environmental clauses in CNF format

## Requirements

- Python 3.8+
- Z3 SMT solver
- NumPy
- Pandas

Install all dependencies:

```bash
pip install -r requirements.txt