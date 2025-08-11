# PAREFT Scaffold
![tests](https://github.com/anulum/PAREFT/actions/workflows/python-tests.yml/badge.svg)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python - <<'PY'
from pareft.config import load_config
from pareft.simulate import run_experiment
cfg = load_config('configs/example.yaml')
run_experiment(cfg)
PY
python scripts/analyze.py outputs/example --show