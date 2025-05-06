# CoreTIA
<span style="font-size: 1.4em; font-weight: bold;">a modular, statistically robust transduction inhibition assay for AAV neutralization</span>

Our methodology is described in detail in our preprint:
> Kovács, B et al. (2025). CoreTIA: A Modular, Statistically Robust Approach for Quantifying AAV Neutralization. *bioRxiv*. [doi:10.1101/2025.04.30.651383v1](https://www.biorxiv.org/content/10.1101/2025.04.30.651383v1)

## Installing

### Installing `uv` package manager

`uv` is a fast Python package installer and resolver, highly recommended. If you don't have it installed, see the [`uv` documentation](https://github.com/astral-sh/uv#installation).

### Installing CoreTIA using `uv`

It's recommended to install CoreTIA in a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
uv venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

## Tutorials
To get started with using CoreTIA, see the tutorials available in the [Tutorials](coretia/tutorials) directory.

## Re-creating figures of the paper
```bash
cd coretia/data
python generate_all.py
```
