import numpy as np
import pandas as pd

from src.stats.stationarity import (
    adf_test, 
    kpss_test, 
    pp_test, 
    run_stationarity_tests)

np.random.seed(42)
n = 500

# Stationary series: AR(1) with |phi| < 1
phi = 0.6
eps = np.random.normal(0, 1, n)
stationary = np.zeros(n)
for t in range(1, n):
    stationary[t] = phi * stationary[t - 1] + eps[t]
stationary_series = pd.Series(stationary, name="stationary_ar1")

# Non-stationary series: random walk with drift
drift = 0.2
innov = np.random.normal(0, 1, n)
nonstationary = np.cumsum(drift + innov)
nonstationary_series = pd.Series(nonstationary, name="nonstationary_random_walk")

print("Stationary series tests:")
for test in [adf_test, kpss_test, pp_test]:
    result = test(stationary_series)
    print(f"{result.test_name} test: p-value={result.p_value:.4f}, is_stationary={result.is_stationary}")

print("\nNon-stationary series tests:")
for test in [adf_test, kpss_test, pp_test]:
    result = test(nonstationary_series)
    print(f"{result.test_name} test: p-value={result.p_value:.4f}, is_stationary={result.is_stationary}")
    
print("\nFinal verdict:")
print('Voted stationarity for stationary series:', run_stationarity_tests(stationary_series).stationary_vote)
print('Voted stationarity for non-stationary series:', run_stationarity_tests(nonstationary_series).stationary_vote)