# %% [markdown]
# # 01 — Data Quality & Cleaning
#
# Challenge Section 1: Exploratory / Data Quality
#
# This script is the .py mirror of `notebooks/01_data_quality.ipynb` (same cells, in the
# same order — the `# %%` markers make each block runnable individually in VS Code's
# Interactive Window / Jupyter). It:
#
# 1. Loads the raw trip data and gets an overview (shape, dtypes, missingness).
# 2. Checks the five anomaly types called out in the brief:
#      - Negative fares
#      - Trip distance of 0 with a nonzero fare
#      - Passenger counts of 0
#      - Drop-off at/before pickup
#      - Unrealistic speeds (distance / duration)
# 3. For each anomaly: quantifies the % of rows affected, decides drop / impute / filter,
#    and comments the justification directly above the code that applies it.
# 4. Applies the cleaning, summarises the impact, and saves the cleaned dataset for the
#    downstream modelling notebooks.

# %%
import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# %% [markdown]
# ## 1. Load data

# %%
RAW_PATH = "../data/raw/Urban_Flow_Analytics_Taxi_Dataset_2026-01.csv"

# Explicit dtypes avoid the mixed-type inference warning pandas raises on this file
# (offline_record_flag is 'Y'/'N' with a large share of blanks -> nullable string).
DTYPES = {
    "provider_code": "Int64",
    "rider_count": "float64",
    "distance_miles": "float64",
    "rate_class_id": "float64",
    "offline_record_flag": "string",
    "origin_loc_id": "Int64",
    "dest_loc_id": "Int64",
    "fare_settlement_method": "Int64",
    "base_fare": "float64",
    "surcharge_misc": "float64",
    "transit_tax": "float64",
    "driver_tip_payment": "float64",
    "toll_total": "float64",
    "service_improvement_fee": "float64",
    "charge_total": "float64",
    "zone_congestion_fee": "float64",
    "Airport_fee": "float64",
    "congestion_relief_fee": "float64",
}

df = pd.read_csv(
    RAW_PATH,
    dtype=DTYPES,
    parse_dates=["pickup_timestamp", "dropoff_timestamp"],
)

n_raw = len(df)
print(f"Rows: {n_raw:,}  |  Columns: {df.shape[1]}")
df.head()

# %% [markdown]
# ## 2. Overview

# %%
df.dtypes

# %%
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
pd.DataFrame({
    "missing_count": missing,
    "missing_pct": (missing / n_raw * 100).round(2),
})

# %% [markdown]
# Observation: `rider_count`, `rate_class_id`, `offline_record_flag`, `zone_congestion_fee`
# and `Airport_fee` are all missing for the same ~1.09M rows (29.2%). This lines up with which
# `provider_code` supplied the record (some providers, e.g. code 6, do not transmit these
# fields) rather than being random — it is a systematic gap, not noise. These columns aren't
# part of the five named anomaly checks below, so we address them separately here: impute the
# categorical / count fields with the mode and the fee fields with 0 (no fee reported = no fee
# charged), keeping an `_missing` flag so downstream models can still use the missingness as a
# signal.

# %%
df["rider_count_missing"] = df["rider_count"].isna()

df["rider_count"] = df["rider_count"].fillna(df["rider_count"].mode()[0])
df["rate_class_id"] = df["rate_class_id"].fillna(df["rate_class_id"].mode()[0])
df["offline_record_flag"] = df["offline_record_flag"].fillna("N")
df["zone_congestion_fee"] = df["zone_congestion_fee"].fillna(0.0)
df["Airport_fee"] = df["Airport_fee"].fillna(0.0)

df.isna().sum().sum()  # 0 -> no missing values left

# %% [markdown]
# ## 3. Anomaly detection
#
# Each check below reports the count and % of total rows affected, then applies the decided
# treatment. Row-level anomaly flags are kept in `drop_mask` / dedicated boolean columns so the
# final cleaning step (Section 4) can report the combined impact without double-counting rows
# flagged by more than one rule.

# %% [markdown]
# ### 3.1 Negative fares
#
# A negative `base_fare` is not physically meaningful for a completed, charged trip.

# %%
flag_negative_fare = df["base_fare"] < 0
n = flag_negative_fare.sum()
print(f"base_fare < 0: {n:,} rows ({n / n_raw * 100:.3f}% of total)")

# Most negative-fare rows coincide with disputed / no-charge / voided settlement codes,
# i.e. billing adjustments rather than genuine paid trips.
df.loc[flag_negative_fare, "fare_settlement_method"].value_counts()

# %% [markdown]
# **Decision: drop.** Negative fares are invalid for a trip that was actually charged, are
# concentrated in dispute/void/no-charge settlement codes (billing corrections, not real
# fares), and affect only ~1.06% of rows — safe to remove without materially reducing the
# dataset.

# %% [markdown]
# ### 3.2 Trip distance of 0 with a nonzero fare

# %%
flag_zero_dist_fare = (df["distance_miles"] == 0) & (df["base_fare"] > 0)
n = flag_zero_dist_fare.sum()
print(f"distance_miles == 0 & base_fare > 0: {n:,} rows ({n / n_raw * 100:.3f}% of total)")

df.loc[flag_zero_dist_fare, "base_fare"].describe()

# %% [markdown]
# **Decision: drop.** A charged trip with 0 recorded distance means the meter/GPS failed to
# log movement — the record can't be trusted for a fare or distance predictor, and there's no
# reliable way to impute the missing distance. This is the largest single anomaly (~3.29% of
# rows); we drop rather than impute since fabricating distance would bias every distance-based
# feature.

# %% [markdown]
# ### 3.3 Passenger count of 0

# %%
flag_zero_riders = df["rider_count"] == 0
n = flag_zero_riders.sum()
print(f"rider_count == 0: {n:,} rows ({n / n_raw * 100:.3f}% of total)")

# %% [markdown]
# **Decision: impute (mode).** A charged trip implies at least one rider, so 0 is an invalid
# entry rather than a real trip we want to lose. `rider_count` is not a primary predictor for
# the fare/duration/demand tasks in this challenge, so rather than drop ~0.4% of otherwise-
# valid trips we replace 0 with the dataset mode (1 rider) and keep a flag for transparency.

# %%
df["rider_count_was_zero"] = flag_zero_riders
df.loc[flag_zero_riders, "rider_count"] = df["rider_count"].mode()[0]

# %% [markdown]
# ### 3.4 Drop-off at or before pickup
#
# The brief calls out "drop-off before pickup". We check the full non-positive range
# (`dropoff <= pickup`), since a trip with 0 duration but a nonzero distance/fare is equally
# invalid — the meter clock did not register real elapsed time.

# %%
duration_sec = (df["dropoff_timestamp"] - df["pickup_timestamp"]).dt.total_seconds()

flag_bad_duration = duration_sec <= 0
n_before = (duration_sec < 0).sum()
n_zero = (duration_sec == 0).sum()
print(f"dropoff before pickup: {n_before:,} rows ({n_before / n_raw * 100:.3f}%)")
print(f"dropoff == pickup:     {n_zero:,} rows ({n_zero / n_raw * 100:.3f}%)")
print(f"combined <= 0:         {flag_bad_duration.sum():,} rows ({flag_bad_duration.sum() / n_raw * 100:.3f}%)")

# %% [markdown]
# **Decision: drop.** A non-positive trip duration is impossible for a trip that covered real
# distance, and cannot be repaired (we don't know the true pickup/drop-off times). This also
# matters directly for Section 2.2 (trip duration prediction), where a duration target of 0 or
# negative would corrupt training. Affects ~1.21% of rows.

# %% [markdown]
# ### 3.5 Unrealistic speed (distance / duration)

# %%
duration_hr = duration_sec / 3600
speed_mph = df["distance_miles"] / duration_hr.replace(0, np.nan)

SPEED_LIMIT_MPH = 80  # generous upper bound for city + highway taxi travel

flag_high_speed = (speed_mph > SPEED_LIMIT_MPH).fillna(False)
n = flag_high_speed.sum()
print(f"implied speed > {SPEED_LIMIT_MPH} mph: {n:,} rows ({n / n_raw * 100:.3f}% of total)")

speed_mph[flag_high_speed].sort_values(ascending=False).head(10)

# %% [markdown]
# **Decision: filter (drop).** Implied speeds in the hundreds of thousands of mph trace back
# to a handful of `distance_miles` values in the hundreds of thousands (clear unit/sensor
# errors) — not something to impute. A 100 mph cap is generous for taxi travel (including
# highway runs to JFK/Newark/Nassau/Westchester) and only removes ~0.02% of rows.

# %% [markdown]
# ## 4. Apply cleaning

# %%
drop_mask = (
    flag_negative_fare
    | flag_zero_dist_fare
    | flag_bad_duration
    | flag_high_speed
)

df_clean = df.loc[~drop_mask].copy()

# Engineered columns useful for downstream notebooks (fare / duration / demand modelling)
df_clean["trip_duration_min"] = (
    (df_clean["dropoff_timestamp"] - df_clean["pickup_timestamp"]).dt.total_seconds() / 60
)
df_clean["speed_mph"] = df_clean["distance_miles"] / (df_clean["trip_duration_min"] / 60)

n_dropped = drop_mask.sum()
print(f"Dropped: {n_dropped:,} rows ({n_dropped / n_raw * 100:.3f}%)")
print(f"Remaining: {len(df_clean):,} rows ({len(df_clean) / n_raw * 100:.3f}%)")

# %% [markdown]
# ## 5. Summary of anomalies and actions

# %%
summary = pd.DataFrame([
    {
        "anomaly": "Negative base_fare",
        "rows_affected": int(flag_negative_fare.sum()),
        "pct_of_total": round(flag_negative_fare.sum() / n_raw * 100, 3),
        "action": "drop",
    },
    {
        "anomaly": "distance_miles == 0 & base_fare > 0",
        "rows_affected": int(flag_zero_dist_fare.sum()),
        "pct_of_total": round(flag_zero_dist_fare.sum() / n_raw * 100, 3),
        "action": "drop",
    },
    {
        "anomaly": "rider_count == 0",
        "rows_affected": int(flag_zero_riders.sum()),
        "pct_of_total": round(flag_zero_riders.sum() / n_raw * 100, 3),
        "action": "impute (mode)",
    },
    {
        "anomaly": "dropoff_timestamp <= pickup_timestamp",
        "rows_affected": int(flag_bad_duration.sum()),
        "pct_of_total": round(flag_bad_duration.sum() / n_raw * 100, 3),
        "action": "drop",
    },
    {
        "anomaly": f"implied speed > {SPEED_LIMIT_MPH} mph",
        "rows_affected": int(flag_high_speed.sum()),
        "pct_of_total": round(flag_high_speed.sum() / n_raw * 100, 3),
        "action": "filter (drop)",
    },
    {
        "anomaly": "Combined rows removed (union, no double-count)",
        "rows_affected": int(n_dropped),
        "pct_of_total": round(n_dropped / n_raw * 100, 3),
        "action": "-",
    },
])
summary

# %% [markdown]
# ## 6. Quick visual check (before vs. after)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].boxplot([df["base_fare"], df_clean["base_fare"]], tick_labels=["raw", "clean"], showfliers=False)
axes[0].set_title("base_fare")

axes[1].boxplot([df["distance_miles"], df_clean["distance_miles"]], tick_labels=["raw", "clean"], showfliers=False)
axes[1].set_title("distance_miles")

axes[2].boxplot([df_clean["trip_duration_min"]], tick_labels=["clean"], showfliers=False)
axes[2].set_title("trip_duration_min (clean only)")

fig.suptitle("Key fields before vs. after cleaning (outliers hidden)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Save cleaned dataset

# %%
OUT_PATH = "../data/processed/taxi_trips_clean.csv"
df_clean.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df_clean):,} rows to {OUT_PATH}")

# %% [markdown]
# ## Conclusion
#
# - Five anomaly types were checked; combined they affect ~5.55% of rows, all dropped except
#   `rider_count == 0` (~0.40%), which was imputed since it doesn't warrant losing an
#   otherwise valid trip record.
# - A separate, systematic missing-data pattern (~29.2% of rows, tied to `provider_code`) was
#   imputed with sensible defaults and flagged via `rider_count_missing`.
# - The cleaned dataset (`data/processed/taxi_trips_clean.csv`) retains ~94.45% of the
#   original rows and is the input for the modelling notebooks (fare prediction, duration
#   prediction, demand forecasting, hotspot clustering).
