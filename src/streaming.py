import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, field
import time

from src.config import DATA_DIR, RESULTS_DIR

# Constants
JUMP_INTERVAL_S = 10  # p_k sensors jump every 10 seconds
HALF_LIFE_S = 5       # Decay half-life
DECAY_LAMBDA = np.log(2) / HALF_LIFE_S
SNAPSHOT_INTERVAL_S = 30


@dataclass
class JumpDecaySensor:
    """Sensor with jump/decay behavior (p_k sensors)."""
    k: int
    value: float = 0.0
    last_jump_time_s: float = 0.0
    last_jump_value: float = 0.0
    
    def update(self, current_time_s: float, rng: np.random.Generator) -> float:
        """Update sensor value with jump/decay logic.
        
        Every 10 seconds: 50% chance +k, 50% chance -k
        Between jumps: exponential decay with half-life 5s
        """
        # Check if time for new jump - every 10 seconds
        time_since_last_jump = current_time_s - self.last_jump_time_s
        
        if time_since_last_jump >= JUMP_INTERVAL_S:
            # Time for a jump
            jump_direction = 1 if rng.random() < 0.5 else -1
            jump_amount = jump_direction * self.k
            
            # Apply decay to current value first, then add jump
            decayed_value = self.last_jump_value * np.exp(-DECAY_LAMBDA * time_since_last_jump)
            self.value = decayed_value + jump_amount
            
            # Record jump
            self.last_jump_time_s = current_time_s
            self.last_jump_value = self.value
        else:
            # Just decay from last jump
            self.value = self.last_jump_value * np.exp(-DECAY_LAMBDA * time_since_last_jump)
        
        return self.value


def load_historical_data_for_streaming(data_dir, num_days):
    partitioning = ds.partitioning(
        pa.schema([("product_id", pa.uint8())]),
        flavor="hive"
    )
    
    dataset = ds.dataset(data_dir, format="parquet", partitioning=partitioning)
    
    # Calculate end timestamp (first 5 days of 2020)
    start_ns = np.datetime64("2020-01-01", "ns").astype(np.int64)
    end_ns = np.datetime64(f"2020-01-{num_days + 1:02d}", "ns").astype(np.int64)
    
    # Load with time filter
    table = dataset.to_table(
        filter=(ds.field("epoch_ns") >= start_ns) & (ds.field("epoch_ns") < end_ns)
    )
    
    df = table.to_pandas()
    df = df.sort_values("epoch_ns").reset_index(drop=True)
    
    return df


def run_streaming_simulation(data_dir = DATA_DIR, num_days = 5):
    """Replay historical data and simulate streaming with p_k sensors.
    
    Args:
        data_dir: Path to historical data
        num_days: Number of days to replay (default 5)
    """    
    print("Streaming Simulation")

    # Setup output
    snapshots_dir = RESULTS_DIR / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hist data
    start_time = time.time()
    df = load_historical_data_for_streaming(data_dir, num_days)
    print(f"  Loaded {len(df):,} rows in {time.time() - start_time:.1f}s")
    
    if df.empty:
        print("Error: No data found!")
        return
    
    # Initialize p_k sensors
    rng = np.random.default_rng(seed=42)
    p_sensors = {k: JumpDecaySensor(k=k) for k in range(11)}
    
    # Track latest values for each (product_id, sensor_id)
    # Key: (product_id, sensor_id), Value: (epoch_ns, value)
    latest_values: dict[tuple[int, int], tuple[int, float]] = {}
    
    # Get time range
    start_ns = df["epoch_ns"].min()
    end_ns = df["epoch_ns"].max()
    start_s = start_ns / 1e9
    end_s = end_ns / 1e9
    
    print(f"Time range: {pd.Timestamp(start_ns, unit='ns')} to {pd.Timestamp(end_ns, unit='ns')}")
    
    # Snapshot tracking
    next_snapshot_s = start_s + SNAPSHOT_INTERVAL_S
    snapshot_count = 0
    all_snapshots = []
    
    # to numpy 
    epochs = df["epoch_ns"].values
    products = df["product_id"].values
    sensors = df["sensor_id"].values
    values = df["value"].values
    
    
    sim_start = time.time()
    
    # Process each event in order
    for i in range(len(df)):
        epoch_ns = epochs[i]
        current_s = epoch_ns / 1e9
        
        # Update latest value
        key = (int(products[i]), int(sensors[i]))
        latest_values[key] = (epoch_ns, float(values[i]))
        
        # Update p_k sensors
        for k, sensor in p_sensors.items():
            sensor.update(current_s, rng)
        
        # Check if snapshot time
        while current_s >= next_snapshot_s:
            snapshot = take_snapshot(next_snapshot_s, latest_values, p_sensors)
            all_snapshots.append(snapshot)
            snapshot_count += 1
            
            if snapshot_count % 100 == 0:
                print(f"    Snapshots: {snapshot_count}")
            
            next_snapshot_s += SNAPSHOT_INTERVAL_S
    
    sim_time = time.time() - sim_start
    print(f"Simulation complete in {sim_time:.1f}s")
    print(f"Total snapshots: {snapshot_count}")
    
    # Save snapshots
    if all_snapshots:
        save_snapshots(all_snapshots, snapshots_dir)


def take_snapshot(time_s: float, latest_values: dict, p_sensors: dict) -> dict:
    epoch_ns = int(time_s * 1e9)
    
    snapshot = {
        "snapshot_time_ns": epoch_ns,
        "snapshot_time": pd.Timestamp(epoch_ns, unit="ns"),
    }
    
    # Add field_x sensors 
    for (product_id, sensor_id), (ts, value) in latest_values.items():
        col_name = f"field_{sensor_id}_p{product_id}"
        snapshot[col_name] = value
    
    # Add p_k sensors
    for k, sensor in p_sensors.items():
        snapshot[f"p_{k}"] = sensor.value
    
    return snapshot


def save_snapshots(snapshots: list[dict], output_dir: Path):
    df = pd.DataFrame(snapshots)
    
    # Sort column, snapshot_time first, then p_k, then field_x
    cols = ["snapshot_time_ns", "snapshot_time"]
    p_cols = sorted([c for c in df.columns if c.startswith("p_")])
    field_cols = sorted([c for c in df.columns if c.startswith("field_")])
    cols = cols + p_cols + field_cols
    
    # column that exists
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    output_file = output_dir / "snapshots.parquet"
    df.to_parquet(output_file, compression="zstd")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")