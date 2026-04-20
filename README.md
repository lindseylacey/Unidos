# Unidos Analysis Repo

This repository contains ACS/CPS notebooks for estimating likely unauthorized immigrant populations using a Borjas-style methodology.

## Repository Layout

- `notebooks/immigration/acs_detailed_estimates.ipynb`
  - ACS-based implementation of a Borjas-style methodology to classify foreign-born records as likely legal vs likely unauthorized.
  - Includes chunked IPUMS loading and stepwise rules.

- `notebooks/immigration/cps_detailed_estimates.ipynb`
  - CPS-based version of the same methodology.
  - Applies parallel rule logic to CPS variables.

## Data Inputs

The notebooks currently reference local absolute file paths under `C:\Users\linds\OneDrive\Documents\...`.

Expected source files include examples such as:

- `acs_1.xml`, `acs_1.dat.gz`
- `cps_00007.xml`, `cps_00007.dat.gz`

## Environment

Recommended Python version: 3.10+

Install core dependencies:

```bash
pip install pandas matplotlib openpyxl ipumspy
```

Notes:
- `openpyxl` is required for Excel reads in pandas.
- IPUMS workflows use `ipumspy` and can be memory intensive for full-file reads.

## How to Run

1. Open a notebook in VS Code.
2. Select your Python kernel (ideally from the repo `venv`).
3. Run cells top-to-bottom.
4. Verify columns with quick checks (for example `print(df.columns.tolist())`) before plotting if column names differ by source file version.

## Workflow Summary

### ACS/CPS detailed estimates
- Restrict to working-age sample.
- Restrict to foreign-born population.
- Apply stepwise legal-status proxy rules (immigration year, benefits, veteran status, citizenship, etc.).
- Produce counts and subgroup summaries.

