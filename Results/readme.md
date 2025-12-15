# RL-Based Option Hedging vs Classical Delta Hedging

## 📌 Project Overview

This project studies whether a **model-free Reinforcement Learning (RL) agent (PPO)** can learn an effective option hedging strategy and outperform the **classical Black–Scholes delta hedging baseline** under realistic trading frictions such as transaction costs.

A custom Gym environment simulates option hedging in a **Geometric Brownian Motion (GBM)** market. The RL agent’s performance is evaluated against a mathematically grounded baseline to ensure a fair and interpretable comparison.

---

## 🧠 Baseline Strategy: Delta Hedging

Delta hedging is derived from the **Black–Scholes model**, which provides a closed-form expression for:

* Option price
* Option delta (sensitivity to the underlying asset price)

Under ideal assumptions (constant volatility, continuous trading, no arbitrage), delta hedging is **near-optimal** and widely used in industry.

In this project, delta hedging serves as a **strong low-variance baseline**.

---

## 🤖 RL Strategy: Proximal Policy Optimization (PPO)

The PPO agent:

* Has **no access to analytical finance formulas**
* Observes only market state variables (price, time, hedge position)
* Learns a hedging policy purely through reward feedback

The goal is to evaluate whether **data-driven adaptive control** can match or exceed classical hedging strategies.

---

## 📊 Experimental Results

### Quantitative Summary

```
PPO Mean Reward     : -101.86
PPO Reward Std      : 171.28
--------------------
Delta Mean PnL      : -0.21
Delta PnL Std       : 0.93
Delta Avg Cost      : 0.27
```

> **Note:** PPO rewards and Delta PnL are reported on different scales and should not be directly compared numerically.

---

## 🔍 Key Insights

### 1️⃣ RL fails to outperform the baseline under baseline assumptions

Under clean Black–Scholes assumptions (GBM, constant volatility), the PPO agent underperforms delta hedging. This is a **theoretically consistent outcome**.

> Naive RL struggles to outperform analytically optimal strategies under idealized market models.

---

### 2️⃣ Delta hedging is a strong low-variance baseline

* Mean PnL loss is small
* Variance is low
* Losses are primarily due to transaction costs

This validates:

* Correct baseline implementation
* Realistic market simulation
* Sound experimental setup

---

### 3️⃣ Transaction costs dominate hedging performance

The average transaction cost exceeds the mean PnL loss, indicating that **over-trading is the primary performance bottleneck** even for classical strategies.

---

### 4️⃣ RL is highly sensitive to reward shaping

The PPO agent exhibits high variance and large negative cumulative rewards, highlighting:

* Sensitivity to reward scaling
* Penalty accumulation across timesteps
* Exploration instability

Reward design is often more important than algorithm choice in financial RL.

---

### 5️⃣ RL requires regime complexity to shine

RL methods typically outperform classical hedging only when ideal assumptions are violated, such as:

* Stochastic volatility
* Jump diffusion
* Regime switching
* Liquidity constraints
* Delayed execution

Since the environment closely follows Black–Scholes assumptions, delta hedging remains difficult to beat.

---

## 🧪 Interpretation

These results **do not indicate project failure**. Instead, they validate known principles in quantitative finance:

* Classical hedging is hard to beat under clean assumptions
* RL does not provide free gains without additional realism
* Negative or neutral results are meaningful experimental outcomes

---

## 🚀 Future Work

Potential extensions where RL is expected to outperform classical methods:

* Stochastic volatility models (e.g., Heston)
* Jump processes
* Liquidity-aware cost models
* Execution delays
* Regime-aware reward functions

---

## ✅ Conclusion

This project presents an **honest, end-to-end comparison** between classical delta hedging and RL-based hedging.

While PPO underperforms under baseline assumptions, the results:

* Validate theoretical expectations
* Highlight limitations of naive RL
* Motivate richer market modeling

The project forms a strong foundation for future research in **adaptive and cost-aware financial hedging using reinforcement learning**.
