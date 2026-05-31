# CACRS: Confidence-Aware Causal Re-planning with Skills Memory

This repository provides the code accompanying the manuscript:

**CACRS: Confidence-Aware Causal Re-planning with Skills Memory for Long-Horizon Reinforcement Learning**

CACRS is an LLM-assisted reinforcement learning framework for sparse-reward, long-horizon tasks with ordered dependencies. The method combines:  
1. a confidence-aware predicate-level belief graph,  
2. event-triggered causal re-planning,  
3. attempt-level skills memory, and  
4. a goal-conditioned PPO low-level policy.

The manuscript describes the overall framework in Section 4 and summarizes the training and re-planning flow in Appendix A. In particular, CACRS maintains a Bayesian confidence belief graph and an attempt-level skills memory, while LLM outputs are treated as tentative goals and uncertain dependency hypotheses rather than directly executed plans.【1†source】

---

## 1. Repository Overview

A typical repository layout is as follows:

```text
CACRS/
├── crafter_cars/
│   ├── train.py                  # Main Crafter training entry
│   ├── config.yaml               # General experiment configuration
│   ├── ppo.yaml                  # PPO hyperparameters
│   ├── crafter_env.py            # Crafter environment interface
│   ├── language_model.py         # LLM interface / cached proposal logic
│   ├── encoder.py                # Goal / text encoding utilities
│   ├── parse_utils.py            # Parsing utilities for LLM outputs
│   ├── replay_buffer.py          # Rollout / replay utilities
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # Original project notes, if present
│   ├── ablation/
│   │   ├── full_cars.yaml        # Full CACRS setting
│   │   ├── no_bayes.yaml         # Ablation without confidence belief graph
│   │   ├── no_memory.yaml        # Ablation without skills memory
│   │   └── no_replan.yaml        # Ablation without causal re-planning
│   ├── agent/
│   │   ├── algorithm/ppo.py      # PPO update implementation
│   │   ├── model/ppo.py          # Goal-conditioned PPO model
│   │   └── ...                   # Network, sampling, logging, storage modules
│   ├── text_crafter/             # Crafter-like textual/environment components
│   └── wrapper/
│       └── goal_wrapper.py       # Goal-conditioned environment wrapper
│
└── shared/
    ├── belief_graph.py           # Confidence-aware Bayesian predicate graph
    ├── planner.py                # Causal re-planning and goal scoring
    ├── skill_memory.py           # Attempt-level skills memory
    ├── failure_memory.py         # Failed attempt / failure statistics
    ├── llm_proposer.py           # LLM-based candidate goal / edge proposal
    ├── predicates.py             # Predicate representation and utilities
    ├── crafter_adapter.py        # Crafter-specific predicate adapter
    ├── base_adapter.py           # Abstract adapter interface
    └── semantic_shaper.py        # Optional semantic shaping utilities
```

---

## 2. Method-to-Code Mapping

| Manuscript component | Description in the paper | Main implementation files |
|---|---|---|
| Confidence belief graph | CACRS represents task structure as a directed predicate graph whose edges are uncertain dependency hypotheses. Each edge maintains a Beta posterior and a confidence estimate. The posterior is updated with grounded evidence from completed attempts or local dependency checks.【1†source】 | `shared/belief_graph.py`, `shared/predicates.py`, `shared/crafter_adapter.py` |
| Goal proposal and frontier construction | Candidate goals are constructed from causal frontier predicates, retrieved memory goals, and LLM proposals. Already satisfied goals are removed before selection.【1†source】 | `shared/planner.py`, `shared/llm_proposer.py`, `crafter_cars/language_model.py` |
| Reliability-aware causal re-planning | For each candidate goal, CACRS computes causal support, state applicability, attempt-level reliability, risk, value, cost, and uncertainty before selecting the goal with the maximum score.【1†source】 | `shared/planner.py`, `shared/belief_graph.py`, `shared/failure_memory.py` |
| Event-triggered re-planning | CACRS does not re-plan at every step. A new planning decision is triggered after goal completion, stalled progress, unreliable support, or repeated failure.【1†source】 | `shared/planner.py`, `crafter_cars/train.py` |
| Skills memory | The memory stores finalized goal attempts as predicate-grounded skill records. It provides reusable evidence and value bias to the planner, while the low-level policy still performs execution.【1†source】 | `shared/skill_memory.py`, `shared/planner.py` |
| Low-level execution | The selected high-level goal conditions a PPO policy that receives observations and goal information.【1†source】 | `crafter_cars/agent/algorithm/ppo.py`, `crafter_cars/agent/model/ppo.py`, `crafter_cars/wrapper/goal_wrapper.py` |
| Overall training loop | Algorithm A.1 collects transitions, extracts evidence, updates the confidence graph, finalizes attempts, updates memory/failure records, re-plans when needed, and updates PPO using the rollout buffer.【1†source】 | `crafter_cars/train.py`, `shared/*.py`, `crafter_cars/agent/*` |

---

## 3. Installation

### 3.1 Recommended environment

```bash
conda create -n cacrs python=3.9 -y
conda activate cacrs
```

### 3.2 Install dependencies

From the repository root:

```bash
cd CACRS/crafter_cars
pip install -r requirements.txt
```

If the environment package requires additional local installation, install it in editable mode from the repository root:

```bash
pip install -e .
```

> Note: exact package versions should follow `crafter_cars/requirements.txt`. Reviewers are encouraged to report the Python, PyTorch, CUDA, and operating-system versions used for reproduction.

---

## 4. LLM and Embedding Configuration

The manuscript uses **Qwen2.5-7B-Instruct** as the proposal-generating LLM and **all-MiniLM-L6-v2** for text embedding.【1†source】

The LLM is used for high-level proposal generation, not as a step-wise controller. It is queried when internal evidence is insufficient, such as at cold start, unresolved frontier construction, repeated failures, or uncertain dependency support.【1†source】

Depending on the submitted code package, LLM proposals may be generated online or loaded from a local cache such as:

```text
crafter_cars/lm_cache.pkl
```

For anonymous review or offline reproduction, cached LLM outputs can be used to avoid dependence on external API availability.

---

## 5. Running CACRS on Crafter

From the Crafter code directory:

```bash
cd CACRS/crafter_cars
python train.py --config config.yaml
```

If the code uses ablation-specific YAML files, the full CACRS setting can be launched with:

```bash
python train.py --config ablation/full_cars.yaml
```

The manuscript evaluates Crafter under a 1M-step setting and reports environment reward, achievement success rate, and the standard Crafter score.【1†source】

---

## 6. Ablation Experiments

The ablation configuration files correspond to the main component analysis reported in the manuscript:

```bash
# Full CACRS
python train.py --config ablation/full_cars.yaml

# Without confidence belief graph / Bayesian belief component
python train.py --config ablation/no_bayes.yaml

# Without skills memory
python train.py --config ablation/no_memory.yaml

# Without causal re-planning
python train.py --config ablation/no_replan.yaml
```

The manuscript reports that removing causal re-planning produces the largest performance drop, while the confidence belief graph and skills memory provide complementary benefits.【1†source】

---

## 7. Expected Outputs and Logs

A typical run should save or print:

- episodic reward,
- Crafter achievement success rates,
- aggregate Crafter score,
- current selected high-level goal,
- re-planning events,
- belief-graph updates,
- skill-memory updates,
- PPO training statistics.

For reviewer convenience, we recommend preserving the following artifacts for each run:

```text
logs/
├── config.yaml
├── metrics.csv or metrics.json
├── achievement_success.csv
├── planner_events.jsonl
├── belief_graph_snapshot.pkl or .json
├── skill_memory_snapshot.pkl or .json
└── checkpoints/
```

---

## 8. Reproducibility Notes

To reproduce the reported results as closely as possible:

1. Use the same random seeds as those reported in the experiment scripts or configuration files.
2. Keep the training budget consistent with the manuscript, especially the 1M-step Crafter setting.【1†source】
3. Use cached LLM outputs if online LLM calls are unavailable or if the review process requires deterministic reproduction.
4. Keep the same PPO hyperparameters specified in `ppo.yaml`.
5. Use the same ablation configuration files when reproducing Table 2-style component comparisons.

---

## 9. Key Implementation Principles

The code should be interpreted according to the following principles from the paper:

### 9.1 LLM proposals are not directly executed

The LLM proposes tentative goals and dependency hypotheses. These suggestions are inserted into the belief graph with uncertain priors and must be grounded through environment interaction.【1†source】

### 9.2 The belief graph makes the final high-level decision

The planner evaluates candidate goals with grounded evidence, including causal support, current-state applicability, reliability, risk, value, cost, and uncertainty.【1†source】

### 9.3 Re-planning is event-triggered

The active goal is retained while it remains plausible and progress continues. Re-planning occurs after meaningful execution events such as completion, stagnation, unreliable support, or repeated failure.【1†source】

### 9.4 Skills memory stores finalized attempts

Skill records are created at the goal-attempt level rather than at every primitive action step. This avoids incorrectly treating intermediate actions in a long-horizon attempt as independent failures.【1†source】

### 9.5 PPO remains the low-level executor

Skills memory and causal planning bias high-level goal selection; they do not replace the goal-conditioned PPO policy used for low-level control.【1†source】

---

## 10. Reviewer Checklist

Reviewers can inspect the following correspondences:

- `shared/belief_graph.py`: Beta posterior confidence update for predicate dependencies.
- `shared/planner.py`: candidate goal construction, event-triggered re-planning, and goal scoring.
- `shared/skill_memory.py`: finalized attempt storage and predicate-grounded skill relation construction.
- `shared/llm_proposer.py` and `crafter_cars/language_model.py`: LLM candidate goal and dependency proposal.
- `crafter_cars/train.py`: Algorithm A.1-style training loop.
- `crafter_cars/agent/algorithm/ppo.py`: PPO policy optimization.
- `crafter_cars/ablation/*.yaml`: component ablations.

---

## 11. License

Please refer to the `LICENSE` file included in the repository.

---

## 12. Contact

For questions about reproduction, please contact the corresponding author listed in the manuscript.
