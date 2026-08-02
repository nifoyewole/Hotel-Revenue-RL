# Hotel-Revenue-RL

Dynamic pricing for hotel revenue management, formulated as a finite-horizon Markov
decision process and solved three ways: exact **Value Iteration**, model-free **Double
Q-learning**, and a **mixed-integer program** for the open-loop rate schedule.

Dataset: [Hotel Booking Demand](https://doi.org/10.1016/j.dib.2018.11.126) (Antonio,
Almeida & Nunes, 2019) — 119,390 bookings at a Portuguese city hotel and resort hotel,
arrivals July 2015 – August 2017.

## The problem

A hotel sells a fixed room inventory over a 12-week booking window. Each week it posts a
rate; the rate determines how much demand converts, demand is random, cancellations erode
booked revenue, and unsold rooms are worthless once the arrival date passes. The task is to
choose rates week by week to maximise season revenue.

**State** `(weeks_remaining, occupancy_bin, market_regime)` — selling time left, inventory
consumed as a decile of capacity, and a latent market condition estimated from historical
demand shocks.
**Action** a price multiplier from `{0.70, 0.85, 1.00, 1.15, 1.30, 1.45}` on the reference
rate.
**Reward** cancellation-discounted revenue less a denied-service cost for turning guests
away once full.
**Dynamics** Poisson arrivals with mean `base[w] · φ[regime] · multiplier^(−ε)`, every term
estimated from the data (see `notebooks/02`).

## Results — City Hotel, May

4,000 simulated seasons per strategy under **common random numbers**, so all strategies face
identical demand realisations and the comparison is paired. *Net revenue* is the quantity
every algorithm optimises: realised revenue less the denied-service cost of turning guests
away once full.

| strategy | net revenue | vs list price | % of optimum | occupancy | realised ADR |
|---|---:|---:|---:|---:|---:|
| **DP optimal** (Value Iteration) | 528,570 | +18.0% | 99.9% | 0.99 | 94.3 |
| **Double Q-learning** | 524,652 | +17.1% | 99.2% | 0.99 | 93.4 |
| MILP schedule | 510,234 | +13.9% | 96.5% | 0.94 | 95.7 |
| best fixed rate (1.15×) | 457,270 | +2.1% | 86.5% | 0.82 | 98.3 |
| naive 1.00× (list price) | 448,074 | — | 84.7% | 1.00 | 79.6 |
| pace heuristic | 443,461 | −1.0% | 83.9% | 1.00 | 78.9 |
| random | 386,329 | −13.8% | 73.1% | 0.98 | 70.0 |

Both learned policies beat list pricing with Cohen's *d* > 2.4 (paired *t*, Holm-corrected).
Applied **unchanged** to all 24 (hotel, month) cells, the Q-learning policy beats list
pricing in 24/24, median lift +29.5%.

Headline findings:

- **Value Iteration becomes exact after exactly 12 sweeps** — one per week of horizon, as
  finite-horizon theory predicts, with the 13th sweep returning a zero residual. Its analytic
  `V*` matches the simulator's Monte-Carlo return to within sampling error, confirming that
  the transition kernel and the simulator describe the same MDP.
- **Double Q-learning reaches 99.2% of the exact optimum** without ever seeing the model
  (98.9% ± 0.3% across five seeds; the table reports the selected run). Valuing the learned
  policy *exactly* rather than by simulation splits the 0.73% shortfall almost evenly:
  **0.32%** is the cost of seeing occupancy only as a decile, **0.42%** is residual learning
  error. Notably, the agent chooses the optimal action in only **41%** of reachable states
  (**59%** of decisions actually taken, weighted by visit frequency) yet still loses under 1%
  of value — `Q*` is flat over wide regions, so a policy heat map badly overstates
  disagreement. Only the value function says what a difference is worth.
- **n-step returns are essential.** One-step bootstrapping plateaus at 92.7% of the optimum
  and swings across seeds; Monte-Carlo returns reach 98.6% and are stable. Under state
  aggregation the Markov property fails and bootstrap error compounds.
- **Optimising against mean demand is not enough.** The MILP's objective (541,964) *exceeds*
  the true optimum because deterministic demand is worth more than random demand of the same
  mean — yet its schedule earns less than either adaptive policy when replayed under
  uncertainty.

## Repository layout

```
data/raw/          hotel_bookings.csv (not tracked; ships with the submission)
data/processed/    clean.csv, written by notebook 01
notebooks/         01–07, run in order
src/               all logic; notebooks import from here and define nothing themselves
tests/             invariants: kernel/simulator agreement, V* dominance, Holm monotonicity
figures/           every figure, exported at 200 dpi
results/           fitted models, policies and the final comparison table
```

| module | contents |
|---|---|
| `config.py` | structural constants shared by every module |
| `demand.py` | estimation of elasticity, market regimes, booking curve, competitor index |
| `environment.py` | the MDP simulator |
| `mdp.py` | explicit transition kernel, Value Iteration, exact policy evaluation |
| `agents.py` | n-step Double Q-learning |
| `lp.py` | mixed-integer rate schedule and capacity sensitivity |
| `policies.py` | baselines: random, fixed, pace heuristic, schedule replay |
| `metrics.py` | paired evaluation, RevPAR/occupancy metrics, Holm-corrected tests |
| `plotting.py` | shared figure style and export |

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Place `hotel_bookings.csv` in `data/raw/`, then run the notebooks **in order** — each
depends on artefacts written by the previous ones:

| notebook | produces | approx. runtime |
|---|---|---|
| `01_eda.ipynb` | `data/processed/clean.csv` | 30 s |
| `02_demand_model.ipynb` | `cancel_model.joblib`, `demand_tables.joblib` | 2 min |
| `03_environment.ipynb` | MDP demonstration and sanity checks | 1 min |
| `04_value_iteration.ipynb` | `vi_policy.npy`, `vi_solution.joblib` | 1 min |
| `05_q_learning.ipynb` | `q_table.npy`, `q_learning_summary.joblib` | 8 min |
| `06_lp_optimisation.ipynb` | `lp_schedule.npy`, `lp_solution.joblib` | 1 min |
| `07_evaluation.ipynb` | `final_comparison.csv`, `evaluation.joblib` | 5 min |

`results/` and `data/` are build outputs and are not tracked; `src/environment.py` raises a
directed error if they are missing rather than failing obscurely.

Once notebooks 01–02 have run, `pytest tests/` (~12 s) checks the invariants that make the
results meaningful: the cancellation curve runs the right way, Value Iteration terminates in
exactly `H` sweeps, the analytic kernel agrees with the simulator, and no strategy exceeds
`V*`.

## Known limitations

**Price elasticity is not identified from observational data.** Hotels raise rates when
demand is strong, so a naive regression of bookings on rate returns a *positive* coefficient
— an upward-sloping demand curve. The willingness-to-pay estimator used instead reads the
price response off the empirical rate distribution, but that dispersion partly reflects
product mix rather than pure price sensitivity, so it likely overstates elasticity. The two
estimators bracket the truth; `notebooks/07` sweeps elasticity from 1.5 to 4.0 and the
ranking of strategies is unchanged throughout.

**The cross-price (competitor) coefficient is weakly identified** — its confidence interval
spans zero — so it is applied as a modest demand shifter rather than treated as established.

**Demand is assumed independent across weeks.** The simulator cannot represent guests who
delay booking in anticipation of a discount. Strategic customer behaviour would reduce the
gains reported here.

## References

- N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets," *Data in Brief*,
  vol. 22, pp. 41–49, 2019.
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press,
  2018.
- H. van Hasselt, "Double Q-learning," in *Advances in Neural Information Processing
  Systems 23*, 2010.
- K. T. Talluri and G. J. van Ryzin, *The Theory and Practice of Revenue Management*.
  Springer, 2004.
- G. Gallego and G. van Ryzin, "Optimal dynamic pricing of inventories with stochastic demand
  over finite horizons," *Management Science*, vol. 40, no. 8, pp. 999–1020, 1994.

## Licence

MIT — see [LICENSE](LICENSE).
