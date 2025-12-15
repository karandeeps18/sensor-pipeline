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

def load_filtered_data(data_dir: Path, sensor_ids: list[int]) -> pd.DataFrame:
    """Load data using predicate pushdown for hive partitioning"""
    
    partition = ds.partitioning(
        pa.schema([
            ("product_id", pa.uint8())]), 
            flavor = "hive"
    )
    
    dataset = ds.dataset(data_dir, format="parquet", partitioning=partition)
    
    # Filter sensor where id % 3 == 0
    table = dataset.to_table(
        filter=ds.field("sensor_id").isin(sensor_ids)
    )
    
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


def run_analysis(sensor_ids, output_suffix):
    """run analysis pipeline
    args: 
        sensor_ids: List of sensor IDs to analyze 
        output_suffix: suffix for output _with_new_sensor
    returns:
        tuple of (perday_df, accross_days_df)
    """
    # get mondays 
    mondays = get_uk_non_holiday_mondays(2020)
    
    # load filtered data 
    df = load_filtered_data(DATA_DIR, sensor_ids)
    
    # filter UK monday windows  
    df = filter_uk_monday_window(df, mondays)
    
    #compute aggregations 
    per_day = compute_per_day_aggregations(df)
    across_day = compute_across_days_aggregations(per_day)
    
    # save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    per_day_file = RESULTS_DIR / f"per_day{output_suffix}.csv"
    across_days_file = RESULTS_DIR / F"acrss_days{output_suffix}.csv"
    
    return per_day, across_day
    
def get_base_sensor_ids():
    return [i for i in range(NUM_BASE_SENSORS) if i % 3== 0]
    
def get_extended_sensor_ids():
    base = get_base_sensor_ids()
    new = [i for i in range(2001, 2021) if i % 3 == 0]
    return base + new 