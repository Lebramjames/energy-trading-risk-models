import numpy as np
import pandas as pd
from src.stats.cointegration import engle_granger_test

import matplotlib.pyplot as plt


np.random.seed(42)

# Generate two cointegrated series
n = 500
x_cointegrated = np.cumsum(np.random.normal(0, 1, n))
y_cointegrated = 2 * x_cointegrated + np.random.normal(0, 1, n)

# Generate two non-cointegrated series
x_noncointegrated = np.cumsum(np.random.normal(0, 1, n))
y_noncointegrated = np.cumsum(np.random.normal(0, 1, n))

# Convert to pandas Series
dates = pd.date_range("2020-01-01", periods=n)
hh_spot_series = pd.Series(y_cointegrated, index=dates, name="Spot")
hh_fut_series = pd.Series(x_cointegrated, index=dates, name="Futures")

# Cointegration test (cointegrated)
res_coint, spread_coint = engle_granger_test(
    y=hh_spot_series,
    x=hh_fut_series,
    trend="c",
    return_spread=True,
)
print("Cointegrated series test result:")
print(res_coint.to_frame())
print("Spread (first 5 rows):")
print(spread_coint.head())
print("\n")

# Cointegration test (non-cointegrated)
hh_spot_series_nc = pd.Series(y_noncointegrated, index=dates, name="Spot_NC")
hh_fut_series_nc = pd.Series(x_noncointegrated, index=dates, name="Futures_NC")

res_noncoint, spread_noncoint = engle_granger_test(
    y=hh_spot_series_nc,
    x=hh_fut_series_nc,
    trend="c",
    return_spread=True,
)
print("Non-cointegrated series test result:")
print(res_noncoint.to_frame())
print("Spread (first 5 rows):")
print(spread_noncoint.head())

# Optional: plot the series and spreads
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(hh_spot_series, label="Spot (Cointegrated)")
plt.plot(hh_fut_series, label="Futures (Cointegrated)")
plt.legend()
plt.title("Cointegrated Series")

plt.subplot(2, 2, 2)
plt.plot(spread_coint)
plt.title("Spread (Cointegrated)")

plt.subplot(2, 2, 3)
plt.plot(hh_spot_series_nc, label="Spot (Non-Cointegrated)")
plt.plot(hh_fut_series_nc, label="Futures (Non-Cointegrated)")
plt.legend()
plt.title("Non-Cointegrated Series")

plt.subplot(2, 2, 4)
plt.plot(spread_noncoint)
plt.title("Spread (Non-Cointegrated)")

plt.tight_layout()
plt.show()

print("##########" * 5)

from src.stats.cointegration import johansen_test

pjm_spot = pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=dates, name="PJM_Spot")
hh_spot = pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=dates, name="HH_Spot")
hh_fut = hh_spot + np.random.normal(0, 1, n)  # Cointegrated with HH_Spot
hh_fut = pd.Series(hh_fut, index=dates, name="HH_Fut")


df = pd.concat([hh_spot, hh_fut, pjm_spot], axis=1)
res = johansen_test(df, det_order=0, k_ar_diff=1)

print(res.to_frame())
print(res.eigenvectors)