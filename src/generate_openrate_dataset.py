import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

sites = ["news", "sports", "finance", "ecommerce", "education"]
campaign_types = ["promo", "alert", "newsletter", "reminder"]
device_os = ["iOS", "Android", "Web"]
segments = ["new", "active", "loyal", "at_risk"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

df = pd.DataFrame({
    "user_id": np.arange(1, n + 1),
    "site": np.random.choice(sites, n),
    "campaign_type": np.random.choice(campaign_types, n, p=[0.3, 0.2, 0.35, 0.15]),
    "device_os": np.random.choice(device_os, n, p=[0.45, 0.45, 0.10]),
    "hour_of_day": np.random.randint(0, 24, n),
    "day_of_week": np.random.choice(days, n),
    "historical_open_rate": np.round(np.random.beta(2, 5, n), 3),
    "historical_push_count": np.random.poisson(12, n),
    "days_since_last_open": np.random.randint(0, 61, n),
    "segment": np.random.choice(segments, n, p=[0.20, 0.45, 0.25, 0.10]),
})

score = (
    df["historical_open_rate"] * 2.5
    - df["days_since_last_open"] * 0.015
    + (df["hour_of_day"].between(7, 10) | df["hour_of_day"].between(18, 21)).astype(int) * 0.25
    + df["segment"].map({"new": -0.1, "active": 0.1, "loyal": 0.25, "at_risk": -0.2})
    + df["campaign_type"].map({"promo": -0.05, "alert": 0.2, "newsletter": 0.15, "reminder": 0.05})
    + np.random.normal(0, 0.25, n)
)

prob = 1 / (1 + np.exp(-score))
df["target_opened"] = (prob > 0.5).astype(int)

df = df[[
    "user_id",
    "site",
    "campaign_type",
    "device_os",
    "hour_of_day",
    "day_of_week",
    "historical_open_rate",
    "historical_push_count",
    "days_since_last_open",
    "segment",
    "target_opened"
]]

df.to_csv("data/raw/openrate.csv", index=False)
print(df.head())
print(df.shape)