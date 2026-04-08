# DAI-CICS
Institutional Design and Dynamic Analysis of Intervention in the Community-based Integrated Care System

## Constraction
```
src/
├── __init__.py
├── config.py          # ScenarioConfig dataclass, get_scenario_config() factory
├── agents.py          # Elder, Provider, Family, LinkWorker, Manager, AIWatcher
├── model.py           # CareNetworkModel (Mesa 3.x), 9 phase/day
├── metrics.py         # M1–M5 milestones, FC-A/B/C
├── nudges.py          # N1–N4, AgentDynEx dynamic reflection
├── run_experiment.py  # CLI runner (--scenario, --days, --seeds, --plot)
└── plots.py           # matplotlib figure
tests/
└── test_simulation.py # 22 tests
results/               # CSV × 8 + PNG × 6
```

## Scenari
```
# A Only Elder
python -m src.run_experiment --scenario A --days 100

# A/B/C  + plot (recommend)
python -m src.run_experiment --scenario all --days 100 --seeds 10 --plot

# A/B/C + ablations + plot (recommend)
python -m src.run_experiment --all-including-ablations --days 100 --seeds 10 --plot

# Individual Scenario
python -m src.run_experiment --scenario C-noN2 --days 100 --seeds 10
python -m src.run_experiment --scenario C-noN3 --days 100 --seeds 10
python -m src.run_experiment --scenario C-onlyL1 --days 100 --seeds 10

```
