import holidays 
import pandas as pd 
import numpy as np 
import pyarrow.parquet as pq
import pyarrow.dataset as ds 
from pathlib import Path 
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo 
import pyarrow as pa 

from src.config import DATA_DIR, RESULTS_DIR, NUM_BASE_SENSORS

# UK Timezone
UK_TZ = ZoneInfo("Europe/London")

# time windows 9:30-12:30
WINDOW_START = time(9, 30)
WINDOW_END = time(12, 30)
BUCKET_MINUTES = 10 

def get_uk_non_holiday_mondays(year: int)-> list[datetime]:
    """Get all non-holiday monday for UK"""
    uk_holidays = holidays.UK(years=year)
    mondays = []
    date = datetime(year, 1, 1)
    
    # first monday 
    while date.weekday() != 0:
        date += timedelta(days=1)
    
    # all non-holidays mondays
    while date.year == year:
        if date not in uk_holidays:
            mondays.append(date)
        date += timedelta(days=7)
        
    return mondays
    

def get_time_buckets() -> list[tuple[time, time]]:
    """generate 10 min buckets"""
    buckets = []
    current = datetime(2020,1,1, WINDOW_START.hour, WINDOW_START.minute)
    end = datetime(2020,1,1, WINDOW_END.hour, WINDOW_END.minute)
    
    while current < end:
        bucket_start = current.time()
        bucket_end = (current + timedelta(minutes=BUCKET_MINUTES)).time()
        buckets.append((bucket_start, bucket_end))
        current += timedelta(minutes=BUCKET_MINUTES)
    
    return buckets 

def epoch_ns_uk_datetime(epoch_ns: int) -> datetime:
    """convert ns to uk datetime"""
    ts = pd.Timestamp(epoch_ns, unit='ns', tz='UTC')
    return ts.tz_convert(UK_TZ).to_pydatetime()

def get_bucket_label(t: time):
    end = (datetime(2020,1,1, t.hour, t.minute) + timedelta(minutes=BUCKET_MINUTES)).time()
    return f"{t.strftime('%H:%M')}-{end.strftime('%H:%M')}"

def load_filtered_data(
    data_dir: Path,
    sensor_ids: list[int],
    product_id: int | None = None,
) -> pd.DataFrame:
    """
    Load data from a Hive-partitioned Parquet dataset with predicate pushdown.

    - data_dir: root of the dataset (contains product_id=... folders)
    - sensor_ids: numeric sensor IDs to keep
    - product_id: if given, restrict to that product only
    """

    # Hive partitioning on product_id
    partition = ds.partitioning(
        pa.schema([("product_id", pa.uint8())]),
        flavor="hive",
    )

    dataset = ds.dataset(data_dir, format="parquet", partitioning=partition)

    # Build filter
    sensor_filter = ds.field("sensor_id").isin(sensor_ids)

    if product_id is not None:
        product_filter = ds.field("product_id") == product_id
        combined_filter = sensor_filter & product_filter
    else:
        combined_filter = sensor_filter

    table = dataset.to_table(filter=combined_filter)
    return table.to_pandas()

    
def filter_uk_monday_window(df: pd.DataFrame, mondays: list[datetime]) -> pd.DataFrame:
    """Filter data to UK Mondays between 09:30-12:30."""
    # Convert epoch_ns to UK datetime
    df["datetime_uk"] = pd.to_datetime(df["epoch_ns"], unit="ns", utc=True).dt.tz_convert(UK_TZ)
    df["date"] = df["datetime_uk"].dt.date
    df["time"] = df["datetime_uk"].dt.time
    
    # Filter selected Mondays
    monday_dates = {m.date() for m in mondays}
    df = df[df["date"].isin(monday_dates)]
    
    # Filter time
    df = df[(df["time"] >= WINDOW_START) & (df["time"] < WINDOW_END)]
    
    return df
    
def assign_time_bucket(t):
    """Assign a time to its 10-minute bucket."""
    minutes_from_start = (t.hour - WINDOW_START.hour) * 60 + (t.minute - WINDOW_START.minute)
    bucket_idx = minutes_from_start // BUCKET_MINUTES
    bucket_start_minutes = WINDOW_START.hour * 60 + WINDOW_START.minute + bucket_idx * BUCKET_MINUTES
    bucket_start = time(bucket_start_minutes // 60, bucket_start_minutes % 60)
    return get_bucket_label(bucket_start)

def compute_per_day_aggregations(df):
    """Compute mean and std per day, per sensor, per 10-min bucket."""
    df["bucket"] = df["time"].apply(assign_time_bucket)
    
    agg = df.groupby(["date", "sensor_id", "bucket"]).agg(
        mean=("value", "mean"),
        std=("value", "std")
    ).reset_index()
    
    return agg

def compute_across_days_aggregations(per_day_df):
    """Compute mean of means and mean of stds across all days."""
    agg = per_day_df.groupby(["sensor_id", "bucket"]).agg(
        mean=("mean", "mean"),
        std=("std", "mean")
    ).reset_index()
    
    return agg

def run_analysis(sensor_ids: list[int], output_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full analysis pipeline."""
    print(f"Running analysis for {len(sensor_ids)} sensors...")
    
    # Get UK non-holiday Mondays
    mondays = get_uk_non_holiday_mondays(2020)
    print(f"  Found {len(mondays)} non-holiday Mondays in 2020")
    
    # Process one product at a time to save memory
    all_per_day = []
    
    for product_id in range(0, 11):
        print(f"Processing product {product_id}/10")
        
        # Load data for this product only
        df = load_filtered_data(DATA_DIR, sensor_ids, product_id=product_id)
        
        if df.empty:
            print(f"Warning: No data for product {product_id}")
            continue
        
        # Filter to UK Monday windows
        df = filter_uk_monday_window(df, mondays)
        
        if df.empty:
            continue
        
        # Compute per-day aggregations for this product
        per_day = compute_per_day_aggregations(df)
        per_day["product_id"] = product_id
        all_per_day.append(per_day)
        
        # Free memory
        del df
    
    # Combine all products
    if not all_per_day:
        print("  Warning: No data found!")
        return pd.DataFrame(), pd.DataFrame()
    
    per_day = pd.concat(all_per_day, ignore_index=True)
    print(f"  Total per-day rows: {len(per_day):,}")
    
    # Compute across-days aggregations
    print("  Computing across-days aggregations...")
    across_days = compute_across_days_aggregations(per_day)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    per_day_file = RESULTS_DIR / f"per_day{output_suffix}.csv"
    across_days_file = RESULTS_DIR / f"across_days{output_suffix}.csv"
    
    per_day.to_csv(per_day_file, index=False)
    across_days.to_csv(across_days_file, index=False)
    
    print(f"  Saved: {per_day_file}")
    print(f"  Saved: {across_days_file}")
    
    return per_day, across_days

def get_base_sensor_ids():
    return [i for i in range(NUM_BASE_SENSORS) if i % 3== 0]
    
def get_extended_sensor_ids():
    base = get_base_sensor_ids()
    new = [i for i in range(2001, 2021) if i % 3 == 0]
    return base + new 