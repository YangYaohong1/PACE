# PACE

Code for the paper: *An Interactive Framework for Finding the Preferred Trade-off in Differential Privacy*

## What Is Included

- `dp_exp.py`: the standalone experiment script.
- `learning/`: the consolidated preference-learning and Pareto-front update methods, including both importance-sampling and optional MCMC helpers.
- `data/raw/`: the CSV files required by the supported datasets.
- `pace/`: small utilities for dataset cataloging, normalized exports, and launching runs.
- `collect_data.py`: exports normalized dataset views for inspection.
- `run_dp_exp.py`: validates arguments and runs `dp_exp.py` from this folder.

## Install

```bash
cd PACE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, create a Conda environment from the pinned environment file:

```bash
conda env create -f environment.yml
conda activate pace
```

## Run

```bash
python run_dp_exp.py \
  --temperature 0.2 \
  --iterations 20 \
  --repetitions 1 \
  --mode sequential \
  --particles 1000 \
  --cost 1 \
  --dataset adult \
  --interaction curve \
  --curve gompertz
```

Run outputs are written to `results_DP/`, and launch manifests are written to `runs/`.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{yang2026multi,
  title={An Interactive Framework for Finding the Preferred Trade-off in Differential Privacy},
  author={Yaohong Yang, Aki Rehn, Sammie Katt, Antti Honkela and Samuel Kaski},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
