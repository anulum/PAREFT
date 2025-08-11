]633;E;echo '![tests](https://github.com/anulum/PAREFT/actions/workflows/python-tests.yml/badge.svg)';fe30ec88-0207-4ba4-a875-f969f139aaa2]633;C![tests](https://github.com/anulum/PAREFT/actions/workflows/python-tests.yml/badge.svg)

]633;E;echo '![tests](https://github.com/anulum/PAREFT/actions/workflows/python-tests.yml/badge.svg)';60035e90-6489-45b3-81bf-340f3bd231c9]633;C![tests](https://github.com/anulum/PAREFT/actions/workflows/python-tests.yml/badge.svg)

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