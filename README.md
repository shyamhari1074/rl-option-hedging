# Reinforcement Learning for Option Hedging under Transaction Costs

## Overview
This project explores the use of **Reinforcement Learning (RL)** to learn optimal hedging strategies for European options under realistic market conditions.  
Unlike classical delta hedging derived from the Black–Scholes framework, the RL-based approach explicitly accounts for **discrete-time trading, transaction costs, and market uncertainty**.

The objective is **risk minimization**, not speculative profit, aligning with real-world practices in investment banks and derivatives risk management teams.

---

## Motivation
Traditional option hedging approaches rely on strong assumptions:
- Continuous-time trading
- Known and constant volatility
- Zero transaction costs

These assumptions rarely hold in practice.  
This project investigates whether an RL agent can **learn adaptive hedging strategies** that reduce hedging error and transaction costs under more realistic constraints.

---

## Problem Formulation
- **Underlying asset**: Simulated using Geometric Brownian Motion (GBM)
- **Derivative**: European Call Option
- **Objective**: Minimize hedging error and transaction costs over the option’s lifetime

The problem is formulated as a **finite-horizon Markov Decision Process (MDP)**.

---

## Reinforcement Learning Framework

### State Space
The agent observes:
- Current stock price
- Time to maturity
- Current hedge position
- Portfolio value
- (Optional) Black–Scholes delta

### Action Space
- Continuous action representing the hedge position in the underlying asset

### Reward Function
The reward penalizes both risk and trading costs:
\[
r_t = - \left( \text{Hedging Error}_t^2 + \lambda \cdot \text{Transaction Cost}_t \right)
\]

---

## Baseline Strategy
A **Black–Scholes delta hedging strategy** is implemented as a benchmark.

Performance is compared across:
- Hedging error (Mean Squared Error)
- P&L variance
- Transaction costs

---

## Methodology
- Custom Gym-style trading environment
- PPO agent trained using Stable-Baselines3
- Monte Carlo simulation of price paths
- Out-of-sample evaluation under varying volatility regimes

---

## Key Results (Summary)
- The RL agent learns to hedge **less frequently** than delta hedging
- Achieves comparable or lower P&L variance under transaction costs
- Demonstrates robustness under volatility regime changes

---

## Technologies Used
- Python  
- NumPy, SciPy  
- Gymnasium  
- Stable-Baselines3  
- Matplotlib  

---

## Project Structure
rl-option-hedging/
│
├── env/ # Custom hedging environment
├── baselines/ # Delta hedging strategy
├── models/ # RL training scripts
├── evaluation/ # Strategy comparison and metrics
├── results/ # Plots and experiment outputs
└── report/ # Technical report
## References
- Buehler et al., *Deep Hedging*, Quantitative Finance (2019)
- Halperin, *QLBS Model*, Journal of Investment Strategies (2017)

  ## Results
The PPO-based hedging agent significantly reduced total hedging cost compared to an untrained policy and demonstrated adaptive trading behavior under transaction costs. Classical delta hedging exhibited higher transaction costs and PnL variance due to discrete rebalancing.

