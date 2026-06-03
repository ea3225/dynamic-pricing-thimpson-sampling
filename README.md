# Dynamic Pricing using Bayesian Learning and Thompson Sampling

## Overview

This project develops a dynamic pricing strategy for a competitive market with unknown demand. Sellers repeatedly choose prices over multiple periods while learning from observed demand and competitor pricing behavior.

The objective is to maximize expected revenue by combining Bayesian demand estimation with Thompson Sampling, allowing the pricing strategy to continuously adapt to changing market conditions.

The implementation was developed in Python and evaluated in a multi-agent pricing environment where each participant competes against other pricing strategies.

---

## Problem Setting

The environment consists of multiple competing sellers.

At each period:

* Each seller selects a price
* Market demand is realized
* Competitor prices become observable
* Historical demand and pricing information are updated

The challenge is to learn the demand function while simultaneously making revenue-maximizing pricing decisions.

---

## Methodology

### Exploration Phase

To collect informative demand observations, the strategy begins with an exploration stage:

* Initial random price selection
* Structured price cycling across candidate prices
* Continued low-probability exploration throughout the simulation

This prevents premature convergence to suboptimal prices.

### Feature Engineering

Demand is modeled using features that capture both own pricing decisions and the competitive environment.

Features include:

* Own price
* Nonlinear price effects
* Average competitor price
* Minimum competitor price
* Maximum competitor price
* Competitor price dispersion
* Relative price positioning
* Previous demand
* Previous price

These variables provide information about both market conditions and temporal demand dynamics.

### Bayesian Linear Regression

Demand is estimated using Bayesian linear regression.

Key characteristics:

* Gaussian prior on model parameters
* Rolling training window of recent observations
* Recency-weighted learning
* Ridge regularization for numerical stability

The Bayesian framework provides both parameter estimates and uncertainty quantification, which are essential for adaptive decision making.

### Thompson Sampling

Pricing decisions are generated using Thompson Sampling:

1. Sample demand parameters from the posterior distribution
2. Predict demand across a discrete price grid
3. Estimate expected revenue for each candidate price
4. Select the revenue-maximizing price

This naturally balances exploration and exploitation as uncertainty decreases over time.

---

## Implementation Highlights

The pricing strategy includes:

* Bayesian demand learning
* Competitor-aware feature engineering
* Rolling-window model updates
* Recency weighting
* Thompson Sampling over a discretized action space
* Revenue-maximizing price selection

The implementation is contained in:

```text
strategy.py
```

---

## Key Insights

* Modeling competitor behavior significantly improves demand estimation
* Recent observations are more informative than older data in dynamic environments
* Thompson Sampling provides an effective mechanism for balancing learning and revenue generation
* Bayesian methods offer a principled way to incorporate uncertainty into pricing decisions

---

## Repository Structure

```text
dynamic-pricing-thompson-sampling/
│
├── data/
├── strategy.py
├── report/
│   └── final-report.pdf
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
python strategy.py
```

---

## Tech Stack

* Python
* NumPy
* Pandas
* Bayesian Linear Regression
* Thompson Sampling
* Contextual Bandit Methods

---

## Contributors

* Alonso Monroy
* Eleni Apostolou
* Tinna Saemundsdottir

---

## My Contribution

* Developed the Bayesian demand estimation framework using Bayesian linear regression
* Contributed to feature engineering and competitor-price modeling
* Participated in the design and implementation of the Thompson Sampling pricing strategy
* Co-authored the project report, including methodology and experimental design

---

## Report

A detailed description of the methodology and theoretical background is available in:

`report/final-report.pdf`

---

## References

The implementation draws on ideas from:

* Contextual Bandits
* Thompson Sampling
* Bayesian Linear Regression
* Dynamic Pricing under Uncertainty

as discussed in the accompanying project report.

