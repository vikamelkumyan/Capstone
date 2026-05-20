# Capstone Submission Notes

This repository is organized to satisfy the project-files and reproducibility
requirements for the capstone submission.

## Required Artifacts

- Code: `code/train.py`, `code/evaluate.py`, `code/batch_evaluate.py`,
  `code/run_final_training.py`, `code/test_sim.py`, and `code/scripts/`
- Data: `data/raw_data/sumo_data/komitas.net.xml`, `data/raw_data/sumo_data/komitas.sumocfg`,
  `data/raw_data/sumo_data/routes.rou.xml`, plus documented generated route scenarios
- Model: `code/models/komitas_multi_intersection_dqn_ep075.pth`
- Paper/report assets: `paper/main.tex`, `paper/references.bib`, `paper/paper.pdf`,
  and `paper/img/`
- Generated code/result visuals: `code/visualization/`
- Environment: `requirements.txt`
- One-command reproduction: `python code/scripts/reproduce_results.py`

## One-Command Reproduction

Fast reproduction from committed final results:

```bash
python code/scripts/reproduce_results.py
```

This regenerates the training and evaluation figures in
`code/visualization/final_results/` from:

- `code/visualization/final_results/training_log_v3.csv`
- `code/visualization/final_results/eval_ep075_summary.json`

Full simulation rerun:

```bash
python code/scripts/reproduce_results.py --run-evaluation
```

The full rerun requires Eclipse SUMO, `sumo`, and `duarouter` on `PATH`. It
regenerates the scenario route files, runs fixed-time, MaxPressure, and RL
controllers across the reported scenario/seed grid, then regenerates figures.

## Archive Layout

The submission zip should be staged under:

```text
capstone_project/
  paper/
    paper.pdf
    main.tex
    references.bib
    img/
  code/
    train.py
    evaluate.py
    batch_evaluate.py
    run_final_training.py
    test_sim.py
    scripts/
    models/
    visualization/
  data/
    raw_data/
      sumo_data/
    processed_data/
  README.md
  SUBMISSION.md
  requirements.txt
```

It excludes virtual environments, caches, and transient run outputs.
