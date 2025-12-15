""" Synthetic data generation module"""
import math 
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa 
import pyarrow.parquet as pq
from datetime import datetime, timedelta

# Schema as per specs 
SCHEMA = pa.schema([
    ('epoch_ns', pa.int64()),
    ('product_id', pa.uint8()),
    ('sensor_id', pa.int16()),
    ('value', pa.float32()),
    ])

def compute_mu(sensor_id: int, product_id: int) -> float:
    return sensor_id / 10 - 1 + product_id

def compute_sigma(sensor_id: int, product_id: int) -> float:
    return math.log10(sensor_id + 1) + product_id 

def generate_hour_timestamps(hour_start_ns: int, lam: int, min_events: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate possion-distributed timestamps within one hour
    
    args:
        hours_start_ns: start of the hours as nanosec epoch 
        lam: Poisson lamda or E[events per hour]
        min_events: min evnts to generate 
        rng: random generator
        
    returns:
        Array of nanosecond timestamp 
    """
    
    ns_per_hour = 3_600_000_000_000
    
    # min event 
    n_events = max(min_events, rng.poisson(lam))
    
    # uniformly distributed events in one hour
    offsets = rng.integers(0, ns_per_hour, size=n_events, dtype=np.int64)
    return hour_start_ns + offsets

def generate_sensor_data(
    product_id: int,
    sensor_id: int,
    start_date: np.datetime64, 
    num_days: int,
    lam: int,
    min_events: int,
    rng: np.random.Generator
    
) -> tuple[np.ndarray, np.ndarray]:
    """Generate timestamps and values for one product-sensor pair """
    
    ns_per_hour = 3_600_000_000_000
    start_ns = start_date.astype("datetime64[ns]").astype(np.int64)
    
    all_timestamps = []
    num_hours = num_days * 24
    
    # Generate timestamps hour by hour
    for h in range(num_hours):
        hour_start = start_ns + h * ns_per_hour
        ts = generate_hour_timestamps(hour_start, lam, min_events, rng)
        all_timestamps.append(ts)
    
    timestamps = np.concatenate(all_timestamps)
    
    # Generate values from Gaussian distribution
    mu = compute_mu(sensor_id, product_id)
    sigma = compute_sigma(sensor_id, product_id)
    values = rng.normal(mu, sigma, size=len(timestamps)).astype(np.float32)
    
    # Enforce finite values
    non_finite_mask = ~np.isfinite(values)
    while non_finite_mask.any():
        values[non_finite_mask] = rng.normal(mu, sigma, size=non_finite_mask.sum()).astype(np.float32)
        non_finite_mask = ~np.isfinite(values)
    
    return timestamps, values

def generate_batch(
    product_id: int,
    sensor_ids: np.ndarray,
    start_date: np.datetime64,
    num_days: int,
    lam: float,
    min_events: int,
    rng: np.random.Generator
) -> pa.Table:
    """Generate a batch of sensor readings for one product over a time range.
    
    Args:
        product_id: Product ID (1-10)
        sensor_ids: Array of sensor IDs to generate
        start_date: Start date as datetime64[D]
        num_days: Number of days to generate
        lam: Poisson lambda per hour
        min_events: Minimum events per hour
        rng: NumPy random generator
    
    Returns:
        PyArrow Table with columns: epoch_ns, product_id, sensor_id, value
    """
    all_epoch_ns = []
    all_product_id = []
    all_sensor_id = []
    all_value = []
    
    for sid in sensor_ids:
        timestamps, values = generate_sensor_data(
            product_id, sid, start_date, num_days, lam, min_events, rng
        )
        n = len(timestamps)
        
        all_epoch_ns.append(timestamps)
        all_product_id.append(np.full(n, product_id, dtype=np.uint8))
        all_sensor_id.append(np.full(n, sid, dtype=np.int16))
        all_value.append(values)
    
    # Concatenate all arrays
    table = pa.table({
        "epoch_ns": np.concatenate(all_epoch_ns),
        "product_id": np.concatenate(all_product_id),
        "sensor_id": np.concatenate(all_sensor_id),
        "value": np.concatenate(all_value),
    }, schema=SCHEMA)
    
    return table


def write_partition(table: pa.Table, output_dir: Path, product_id: int, batch_idx: int):
    """Write to parquet file partitioned by product_id, using zstd compression"""
    partition_dir = output_dir / f"product_id={product_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = partition_dir / f"batch_{batch_idx:03d}.parquet"
    pq.write_table(table, file_path, compression="zstd")
    
    return file_path