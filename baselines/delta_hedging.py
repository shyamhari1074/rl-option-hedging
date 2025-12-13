from baselines.black_scholes import call_price, delta
from baselines.market_simulator import simulate_gbm


def delta_hedging_episode(
    S0, K, T, r, mu, sigma, steps, transaction_cost, seed=None
):
    prices = simulate_gbm(S0, mu, sigma, T, steps, seed)

    dt = T / steps
    hedge = 0.0
    cash = call_price(S0, K, T, r, sigma)
    total_cost = 0.0

    for t in range(steps):
        tau = T - t * dt
        S = prices[t]

        target_delta = delta(S, K, tau, r, sigma)
        trade = target_delta - hedge

        cost = transaction_cost * abs(trade) * S
        total_cost += cost

        cash -= trade * S + cost
        hedge = target_delta

    payoff = max(prices[-1] - K, 0.0)
    pnl = cash + hedge * prices[-1] - payoff

    return pnl, total_cost
