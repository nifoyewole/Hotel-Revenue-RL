# Hotel-Revenue-RL

Dynamic pricing for hotel revenue management, formulated as a finite-horizon Markov decision process and solved three ways: exact Value Iteration, model-free Double Q-learning, and a mixed-integer program for the open-loop rate schedule.

Dataset: Hotel Booking Demand (Antonio, Almeida and Nunes, 2019) — 119,390 bookings at a Portuguese city hotel and resort hotel, arrivals July 2015 to August 2017.

## The problem

A hotel sells a fixed room inventory over a 12-week booking window. Each week it posts a rate. The rate determines how much demand converts, demand is random, cancellations erode booked revenue, and unsold rooms are worthless once the arrival date passes. The task is to choose rates week by week to maximise season revenue.

### State

Three variables: weeks remaining in the selling window, rooms sold so far as a decile of capacity, and a latent market regime taking values soft, normal or strong. The regime is estimated from historical weekly demand shocks and covers the external-factor requirement; a competitor price index enters as an exogenous demand shifter.

### Action

A price multiplier drawn from 0.70, 0.85, 1.00, 1.15, 1.30, 1.45, applied to the reference rate for that hotel and month.

### Reward

Cancellation-discounted realised revenue, less a denied-service cost for turning guests away once the hotel is full.

### Transitions

Weekly bookings are Poisson with mean equal to the estimated booking curve times the regime demand factor times the multiplier raised to minus the elasticity. Rooms sold accumulate up to capacity, and the regime evolves under a transition matrix counted from the data. Every term is estimated from the historical bookings rather than assumed; notebook 02 does the estimation.

### Constraints

One rate per week, the rate confined to 0.70 to 1.45 times the reference, and cumulative sales bounded by capacity. The horizon is finite and short, so revenue is undiscounted.

## Results

City Hotel, May. Four thousand simulated seasons per strategy under common random numbers, so every strategy faces identical demand realisations and the comparison is paired. Net revenue is the quantity all three methods optimise: realised revenue less the denied-service cost.

Ranked by net revenue, with lift over list pricing and share of the exact optimum:

DP optimal (Value Iteration) — 528,570, plus 18.0 percent, 99.9 percent of optimum, 99 percent occupancy, 94.3 realised ADR.

Double Q-learning — 524,652, plus 17.1 percent, 99.2 percent of optimum, 99 percent occupancy, 93.4 realised ADR.

MILP schedule — 510,234, plus 13.9 percent, 96.5 percent of optimum, 94 percent occupancy, 95.7 realised ADR.

Best fixed rate at 1.15x — 457,270, plus 2.1 percent, 86.5 percent of optimum, 82 percent occupancy, 98.3 realised ADR.

Naive list price at 1.00x — 448,074, the baseline, 84.7 percent of optimum, 100 percent occupancy, 79.6 realised ADR.

Pace heuristic — 443,461, minus 1.0 percent, 83.9 percent of optimum, 100 percent occupancy, 78.9 realised ADR.

Random — 386,329, minus 13.8 percent, 73.1 percent of optimum, 98 percent occupancy, 70.0 realised ADR.

Both learned policies beat list pricing with Cohen's d above 2.4 under paired t-tests with Holm correction. Applied unchanged to all 24 hotel-month cells, the Q-learning policy beats list pricing in 24 of 24, median lift 29.5 percent.

The full table including gross revenue and RevPAR is in results/final_comparison.csv.

## Key findings

Value Iteration becomes exact after exactly 12 sweeps, one per week of horizon, with the thirteenth returning a zero residual. This is what finite-horizon theory predicts, and notebook 04 checks it rather than asserting it. The analytic optimum also matches the simulator's Monte-Carlo return to within sampling error, which confirms that the transition kernel and the simulator describe the same MDP.

Double Q-learning reaches 99.2 percent of the exact optimum without ever seeing the model, and 98.9 percent plus or minus 0.3 across five seeds. Valuing the learned policy exactly rather than by simulation splits the 0.73 percent shortfall almost evenly: 0.32 points are the price of seeing occupancy only as a decile, 0.42 points are residual learning error. Neither dominates.

The agent chooses the optimal action in only 41 percent of reachable states, or 59 percent of decisions actually taken once weighted by visit frequency, yet still loses under 1 percent of value. The action-value surface is flat over wide regions, so a policy heat map badly overstates disagreement. Only the value function says what a difference is worth.

n-step returns are necessary, not a tuning convenience. One-step bootstrapping plateaus at 92.7 percent of the optimum and swings several points across seeds, because the agent sees occupancy only as a decile, the Markov property fails under that aggregation, and bootstrap error compounds at every backup. Monte-Carlo returns reach 98.6 percent and are stable. Notebook 05 runs the ablation.

Optimising against mean demand is not enough. The MILP objective of 541,964 exceeds the true optimum of 528,836, because deterministic demand is worth more than random demand of the same mean. Yet the schedule it produces earns less than either adaptive policy when replayed under uncertainty. Committing to a rate plan in advance is the expensive part, not the optimisation.

The rule-based pace heuristic lands about 1 percent below list pricing. It reacts to occupancy but never values the inventory it gives up. Being state-dependent is not sufficient; the state has to be used to weigh a trade-off.

## Repository layout

data/raw holds hotel_bookings.csv, which ships with the submission but is not tracked in git. data/processed holds clean.csv, written by notebook 01. notebooks holds the seven notebooks, run in order. src holds all the logic; the notebooks import from it and define nothing themselves. tests holds the invariant checks. figures holds every figure at 200 dpi. results holds fitted models, policies and the final comparison table.

The src modules are: config for structural constants shared by everything else; demand for estimating elasticity, market regimes, the booking curve and the competitor index; environment for the MDP simulator; mdp for the explicit transition kernel, Value Iteration and exact policy evaluation; agents for n-step Double Q-learning; lp for the mixed-integer rate schedule and capacity sensitivity; policies for the baselines; metrics for paired evaluation and significance testing; plotting for shared figure style and export.

## Setup

Create and populate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Register the environment as a Jupyter kernel so notebooks resolve to it deterministically:

    .\.venv\Scripts\python.exe -m ipykernel install --user --name hotel-revenue-rl --display-name "Hotel-Revenue-RL (.venv)"

The notebooks record this kernel by name. In VS Code, if prompted, select "Hotel-Revenue-RL (.venv)". Registering matters because the default kernelspec launches a bare "python" rather than an absolute interpreter path, which resolves unpredictably and can hang on connect.

## Running

Place hotel_bookings.csv in data/raw, then run the notebooks in order. Each depends on artefacts written by the previous ones.

Notebook 01 does the exploratory analysis and writes clean.csv, about 30 seconds. Notebook 02 fits the cancellation model and estimates the demand primitives, about 2 minutes. Notebook 03 demonstrates the MDP and runs sanity checks, about 1 minute. Notebook 04 solves the model exactly by Value Iteration, about 1 minute. Notebook 05 trains the Q-learning agent and runs the n-step ablation over five seeds, about 10 minutes. Notebook 06 solves the MILP, about 1 minute. Notebook 07 runs the full evaluation, about 5 minutes.

To run everything headlessly from PowerShell:

    foreach ($nb in "01_eda","02_demand_model","03_environment","04_value_iteration","05_q_learning","06_lp_optimisation","07_evaluation") {
      .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.kernel_name=hotel-revenue-rl --ExecutePreprocessor.timeout=3600 "notebooks\$nb.ipynb"
    }

The results and data directories are build outputs and are not tracked. If they are missing, src/environment.py raises a directed error naming the notebooks to run rather than failing obscurely.

## Testing

Once notebooks 01 and 02 have run:

    .\.venv\Scripts\python.exe -m pytest tests/ -q

Nine tests, about 15 seconds. They check the invariants that make the results meaningful: that cancellation risk falls as arrival approaches, that capacity binds at list price, that the regime chain is stochastic and persistent, that every emitted state indexes the tabular agents safely, that Value Iteration terminates in exactly one sweep per week of horizon, that the analytic kernel agrees with the simulator, that no strategy exceeds the proven optimum, that common random numbers reproduce, and that the Holm correction is monotone and conservative.

## Limitations

Price elasticity is not identified from observational data. Hotels raise rates when demand is strong, so a naive log-log regression of bookings on rate returns a positive coefficient, which would mean an upward-sloping demand curve. The willingness-to-pay estimator used instead reads the price response off the empirical rate distribution, but that dispersion partly reflects product mix rather than pure price sensitivity, so it likely overstates elasticity. The two estimators bracket the truth. Notebook 07 sweeps elasticity from 1.5 to 4.0 and the ranking of strategies is unchanged throughout.

The constant-elasticity functional form is an approximation. It tracks the data closely from list price upward, which is where every optimal policy in this study operates, but overshoots below list price, predicting that more than the whole market would book at a deep discount. Deep discounting is therefore modelled optimistically, which if anything understates how much the learned policies gain by not discounting.

The cross-price competitor coefficient is weakly identified, with a confidence interval spanning zero, so it is applied as a modest demand shifter rather than treated as established.

Demand is assumed independent across weeks. The simulator cannot represent guests who delay booking in anticipation of a discount, and such strategic customer behaviour would reduce the gains reported here.

## References

N. Antonio, A. de Almeida and L. Nunes, "Hotel booking demand datasets," Data in Brief, vol. 22, pp. 41-49, 2019.

R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: MIT Press, 2018.

H. van Hasselt, "Double Q-learning," in Advances in Neural Information Processing Systems 23, 2010, pp. 2613-2621.

K. T. Talluri and G. J. van Ryzin, The Theory and Practice of Revenue Management. New York: Springer, 2004.

G. Gallego and G. van Ryzin, "Optimal dynamic pricing of inventories with stochastic demand over finite horizons," Management Science, vol. 40, no. 8, pp. 999-1020, 1994.

## Licence

MIT. See LICENSE.
