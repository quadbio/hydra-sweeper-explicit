# hydra-sweeper-explicit

[![Tests][badge-tests]][tests]
[![PyPI][badge-pypi]][pypi]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/hydra-sweeper-explicit/test.yaml?branch=main&label=tests
[badge-pypi]: https://img.shields.io/pypi/v/hydra-sweeper-explicit

[tests]: https://github.com/quadbio/hydra-sweeper-explicit/actions/workflows/test.yaml
[pypi]: https://pypi.org/project/hydra-sweeper-explicit

A Hydra sweeper for running explicit parameter combinations without Cartesian product.

## Installation

```bash
pip install hydra-sweeper-explicit
```

## Usage

```yaml
hydra:
  sweeper:
    _target_: hydra_sweeper_explicit.ExplicitSweeper
    combinations:
      - {model: small, lr: 0.01}
      - {model: large, lr: 0.001}
      - {model: large, lr: 0.0001, dropout: 0.5}
```

```bash
python train.py --multirun
```

Runs exactly 3 jobs—no Cartesian product.
