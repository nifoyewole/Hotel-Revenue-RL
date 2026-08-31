# Hotel Revenue Management with Reinforcement Learning

Dynamic hotel room pricing modelled as a finite-horizon Markov Decision Process and solved three ways: Value Iteration (model-based, exact), Double Q-learning (model-free), and Mixed Integer Linear Programming (a fixed weekly rate schedule).

Data: Hotel Booking Demand (Antonio, Almeida and Nunes, 2019), 119,390 bookings from a Portuguese city hotel and resort hotel, arrivals July 2015 to August 2017.

## Where to start
The seven notebooks in `notebooks/` are the submission with all outputs, figures and printed results in place, so the whole project can be read straight through without installing or running anything.

Suggested order, which is also the dependency order:
- 01_eda - data quality, ADR, cancellations, the booking curve
- 02_demand_model - estimating everything the simulator needs from the data
- 03_environment - the MDP itself, plus checks that the simulator behaves
- 04_value_iteration - Algorithm 1, exact optimal policy and the benchmark V*
- 05_q_learning - Algorithm 2, learning the policy from experience alone
- 06_lp_optimisation - the MILP rate schedule and capacity sensitivity
- 07_evaluation - comparison, significance, robustness

If time is short, 03, 04, 05 and 07 carry the core of the argument. Section 6 of notebook 07 is the summary of findings.  
Supporting code lives in `src/`. The notebooks import from it rather than defining methods inline, so `src/mdp.py`, `src/agents.py` and `src/lp.py` are where the three algorithms are actually implemented.


## The problem
A hotel sells a fixed room inventory over a 12-week booking window. Each week it sets a price. Demand is uncertain, roughly 37% of bookings cancel, and any room still unsold at arrival is worth nothing. The goal is to maximise expected revenue over the season.

MDP formulation (defined in `src/environment.py`, described in notebook 03):
- State - weeks remaining (12), occupancy bin (10), market regime (3)
- Action - one of six price multipliers: 0.70, 0.85, 1.00, 1.15, 1.30, 1.45 applied to the hotel's estimated reference rate
- Reward - expected revenue net of cancellations, less a penalty for turned-away demand
- Horizon - 12 weekly decisions, undiscounted (gamma = 1), terminal value zero

Everything the environment needs is estimated from the data in notebook 02: the booking curve, a calibrated cancellation model, price elasticity, market regime transitions, reference rates and a competitor price index. Two estimation problems are worth noting because they are handled explicitly rather than ignored:
- The naive log-log elasticity regression returns the wrong sign, because hotels raise prices when demand is already high. Notebook 02 section 2 shows this and works around the endogeneity.
- The cancellation probability multiplies revenue in the reward, so it has to be calibrated and not merely well ranked. The calibration curve is checked in notebook 02.


## Results
Evaluation setting: City Hotel, May. 4,000 simulated seasons, common random numbers, so every strategy faces the identical sequence of demand and market conditions. Full table with 95% CIs, RevPAR, ADR, occupancy and sell-out rate: `results/final_comparison.csv`.


### Main Findings
* **Double Q-learning achieves 99.2% of the exact optimum** without using the transition model.
* **Adaptive pricing beats fixed schedules:** The MILP loses performance because it cannot react to changing demand.
* **RL significantly outperforms constant pricing:** Q-learning improves revenue by 14.7% over the best fixed rate.
* **Occupancy alone is not enough:** The pace heuristic fills rooms but earns less than simple list pricing.
* **Learned policies match real revenue management behaviour:** They discount early to build demand, then raise prices as availability decreases.
* **Results are robust:** Performance remains strong across different elasticity assumptions, and the Q-learning policy transfers successfully across all 24 hotel-month scenarios.


## Repository layout
data/raw/ - Original dataset (hotel_bookings.csv)  
data/processed/ - Cleaned data created by notebook 01  
notebooks/ - Experiment notebooks (01-07)  
src/ - Core implementation code  
tests/ - Model and evaluation tests  
figures/ - Generated plots  
results/ - Models, policies, and evaluation outputs  

Most files in `data/processed/` and `results/` are generated outputs. They are recreated when the notebooks are run.  


## Source Modules
config.py - Shared settings and constants  
demand.py - Demand estimation and feature generation  
environment.py - MDP simulation environment  
mdp.py - Transition model, Value Iteration, policy evaluation  
agents.py - n-step Double Q-learning  
lp.py - MILP optimisation model  
policies.py - Baseline pricing strategies  
metrics.py - Evaluation metrics and statistical tests  
plotting.py - Plot generation utilities  


## Reproducing Results
### Setup

Create the environment:  
python -m venv .venv  
.\.venv\Scripts\python.exe -m pip install -r requirements.txt  
.\.venv\Scripts\python.exe -m ipykernel install --user --name hotel-revenue-rl  

Pinned versions are in `requirements.txt`. Select the `hotel-revenue-rl` kernel when running the notebooks.

## Data: 
Put the dataset in data/raw/hotel_bookings.csv. If it is missing, download it from https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand and put it there under that name.  

Run the notebooks in this order:
- 01_eda
- 02_demand_model
- 03_environment
- 04_value_iteration
- 05_q_learning
- 06_lp_optimisation
- 07_evaluation


## Tests
pytest tests/ -q  

The tests validate the assumptions behind the results:
- Demand and cancellation curves are correctly modelled
- Booking curves and market regimes behave correctly
- Simulator states remain within valid bounds
- Value Iteration converges correctly
- The analytical MDP and simulator produce consistent results
- No learned policy exceeds the theoretical optimum
- Evaluation results are reproducible
- Statistical corrections are applied correctly


## Limitations

- Elasticity uncertainty: Price sensitivity is estimated from observational data, so the true causal effect of price is uncertain. Results are tested across a range of elasticity values.
- Simplified demand: The model assumes weekly demand is independent and does not capture customers delaying bookings for discounts.
- Limited competition data: Only two hotels are available, so competitor effects are estimated weakly.
- State simplification: The RL agent uses occupancy bins rather than exact room counts.
- Scope limitations: The model does not include multiple room types, length-of-stay decisions, or overbooking.

Despite these limitations, the strategy ranking remains stable across elasticity tests and all hotel-month scenarios.

## AI Use Declaration

Generative AI tools, primarily Claude (Anthropic), were used substantially during the development of this project, particularly for code generation, debugging, refactoring, and explaining implementation approaches. I take responsibility for the final code, analysis, and conclusions presented in this repository.
