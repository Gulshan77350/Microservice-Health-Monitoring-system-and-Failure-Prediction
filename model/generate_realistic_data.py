"""
Realistic synthetic data generator for microservice failure prediction.

WHY THIS EXISTS
----------------
The original dataset (healthy_metrics.csv + failure_metrics.csv) was collected
from two disjoint operating modes of the services: a "healthy" baseline and a
hard-coded "/simulate_failure" state. The two classes never overlap in feature
space (e.g. healthy CPU never exceeds ~80%, failure CPU never drops below ~90%),
so any model -- even a single threshold on error_rate -- gets 100% accuracy.
That's not a model property, it's a data property, and it doesn't reflect how
real production telemetry behaves (noisy, overlapping, probabilistic).

This script generates a new dataset where:
  1. Every feature is drawn from continuous, overlapping distributions.
  2. Failure is NOT a hard threshold -- it's a Bernoulli draw from a logistic
     function of the features, so two services with near-identical metrics
     can have different outcomes (exactly like real systems, where the same
     load sometimes tips a service over and sometimes doesn't, depending on
     factors we don't observe).
  3. Data is generated as a TIME SERIES per service, so a temporal train/test
     split is meaningful (no shuffling across time).

The result is a harder, more realistic problem: expect ~85-93% F1 instead of
100%, a real confusion matrix with both error types, and feature importances
that reflect genuine (if noisy) signal rather than a single dominant feature.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

SERVICES = ["payment", "order", "notification"]
N_DAYS = 10
POINTS_PER_DAY = 480  # every 3 minutes, per service -> manageable size, realistic cadence
START = datetime(2026, 6, 1, 0, 0, 0)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_service_timeseries(service: str, n_points: int) -> pd.DataFrame:
    """
    Generate one service's time series with:
      - a smooth daily load cycle (more load during "business hours")
      - autocorrelated noise (today's load depends on a moment ago, not iid)
      - occasional load spikes that *sometimes* tip into failure and sometimes don't
    """
    t = np.arange(n_points)
    minutes_per_point = (N_DAYS * 24 * 60) / n_points
    hour_of_day = (t * minutes_per_point / 60) % 24

    # Daily seasonality: load peaks around hour 14, trough around hour 4
    daily_cycle = 0.5 + 0.5 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi)

    # Autocorrelated base load via an AR(1)-ish random walk, mean-reverting
    base_load = np.zeros(n_points)
    base_load[0] = 0.4
    for i in range(1, n_points):
        mean_revert = 0.4 + 0.3 * daily_cycle[i]
        base_load[i] = (
            0.85 * base_load[i - 1]
            + 0.15 * mean_revert
            + RNG.normal(0, 0.04)
        )
    base_load = np.clip(base_load, 0.05, 1.0)

    # Inject occasional spike events (incidents) that decay over a few points
    spike = np.zeros(n_points)
    n_spikes = RNG.poisson(N_DAYS * 0.8)  # roughly < 1 spike/day on average
    spike_starts = RNG.choice(n_points, size=min(n_spikes, n_points // 20), replace=False)
    for s in spike_starts:
        duration = RNG.integers(3, 15)
        magnitude = RNG.uniform(0.3, 0.9)
        decay = magnitude * np.exp(-np.arange(duration) / (duration / 2.5))
        end = min(s + duration, n_points)
        spike[s:end] += decay[: end - s]

    load = np.clip(base_load + spike, 0.05, 1.4)  # can exceed 1.0 during real incidents

    # Map load -> raw features with realistic noise, NOT clean thresholds
    cpu = np.clip(20 + load * 65 + RNG.normal(0, 6, n_points), 5, 100)
    memory = np.clip(30 + load * 55 + RNG.normal(0, 5, n_points), 10, 100)
    # latency responds super-linearly to load (queueing effects) + noise
    latency = np.clip(80 + (load ** 2.2) * 1400 + RNG.normal(0, 60, n_points), 30, None)
    requests = np.clip(500 + load * 7000 + RNG.normal(0, 400, n_points), 50, None).astype(int)
    error_rate = np.clip(
        0.2 + (load ** 2.5) * 18 + RNG.normal(0, 1.2, n_points), 0, 60
    )

    # Failure probability: smooth logistic combination of NORMALIZED features,
    # not a hard cutoff. Two points with similar features get similar (not
    # identical) failure probability, and there's irreducible randomness --
    # exactly like a real incident sometimes happening and sometimes not
    # under similar load.
    z = (
        -8.2
        + 0.055 * cpu
        + 0.045 * memory
        + 0.0022 * latency
        + 0.22 * error_rate
        + RNG.normal(0, 0.08, n_points)  # Realistic noise floor
    )
    p_fail = logistic(z)
    failure = RNG.binomial(1, p_fail)

    timestamps = [START + timedelta(minutes=minutes_per_point * i) for i in range(n_points)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "service": service,
            "cpu": np.round(cpu, 2),
            "memory": np.round(memory, 2),
            "latency": np.round(latency, 2),
            "requests": requests,
            "error_rate": np.round(error_rate, 2),
            "failure": failure,
        }
    )


def main():
    frames = []
    n_points = N_DAYS * POINTS_PER_DAY
    for svc in SERVICES:
        frames.append(generate_service_timeseries(svc, n_points))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = "final_dataset_v2.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nFailure distribution:")
    print(df["failure"].value_counts())
    print(f"Failure rate: {df['failure'].mean():.1%}")

    print("\nFeature overlap check (healthy vs failure ranges):")
    for col in ["cpu", "memory", "latency", "requests", "error_rate"]:
        h = df.loc[df.failure == 0, col]
        f = df.loc[df.failure == 1, col]
        overlap = (
            min(h.max(), f.max()) - max(h.min(), f.min())
        )
        print(
            f"  {col:12s} healthy=[{h.min():.1f}, {h.max():.1f}]  "
            f"failure=[{f.min():.1f}, {f.max():.1f}]  "
            f"overlap_width={max(overlap, 0):.1f}"
        )


if __name__ == "__main__":
    main()
