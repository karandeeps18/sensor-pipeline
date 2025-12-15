#!/usr/bin/env python3
"""Script to generate synthetic sensor data."""
import sys
import time
sys.path.insert(0, ".")

import numpy as np

from src.config import (
    YEAR, NUM_BASE_SENSORS, NUM_PRODUCTS,
    LAMBDA_PER_HOUR, MIN_EVENTS_PER_HOUR,
    DATA_DIR, DAYS_PER_BATCH
)
from src.generator import generate_batch, write_partition


def get_month_ranges(year: int) -> list[tuple[np.datetime64, int]]:
    """Get (start_date, num_days) for each month in the year."""
    ranges = []
    for month in range(1, 13):
        start = np.datetime64(f"{year}-{month:02d}-01", "D")
        # Get days in month by going to next month
        if month == 12:
            end = np.datetime64(f"{year + 1}-01-01", "D")
        else:
            end = np.datetime64(f"{year}-{month + 1:02d}-01", "D")
        num_days = int((end - start) / np.timedelta64(1, "D"))
        ranges.append((start, num_days))
    return ranges


def main():
    """Generate sensor data."""
  
    print("IoT Data Generation")
    
    # Setup
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    sensor_ids = np.arange(NUM_BASE_SENSORS, dtype=np.int16)
    month_ranges = get_month_ranges(YEAR)
    
    total_rows = 0
    total_bytes = 0
    start_time = time.time()
    
    print(f"Generating data for {YEAR}")
    print(f"Products: {NUM_PRODUCTS}")
    print(f"Sensors: {NUM_BASE_SENSORS}")
    print(f"Output: {DATA_DIR}")
    
    # Generate data, outer loop by product, inner loop by month
    for product_id in range(0, NUM_PRODUCTS + 1):
        product_start = time.time()
        product_rows = 0
        
        for batch_idx, (start_date, num_days) in enumerate(month_ranges):
            batch_start = time.time()
            
            # Generate batch
            table = generate_batch(
                product_id=product_id,
                sensor_ids=sensor_ids,
                start_date=start_date,
                num_days=num_days,
                lam=LAMBDA_PER_HOUR,
                min_events=MIN_EVENTS_PER_HOUR,
                rng=rng
            )
            
            # Write to disk
            file_path = write_partition(table, DATA_DIR, product_id, batch_idx)
            file_size = file_path.stat().st_size
            
            batch_time = time.time() - batch_start
            product_rows += len(table)
            total_bytes += file_size
            
            print(f"Product {product_id:2d} | {start_date} | "
                  f"rows: {len(table):,} | "
                  f"size: {file_size / 1e6:.1f} MB | "
                  f"time: {batch_time:.1f}s")
        
        product_time = time.time() - product_start
        total_rows += product_rows
        print(f"Product {product_id:2d} complete: {product_rows:,} rows in {product_time:.1f}s")
    
    # Summary
    total_time = time.time() - start_time
    print("Generation Complete")
    print(f"Total rows: {total_rows:,}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Throughput: {total_rows / total_time:,.0f} rows/sec")

    
    # Validation asserts
    assert total_rows > 0, "No data generated"
    assert DATA_DIR.exists(), "Output directory not created"
    
    # Check at least one file per product
    for pid in range(NUM_PRODUCTS):
        partition_dir = DATA_DIR / f"product_id={pid}"
        assert partition_dir.exists(), f"Missing partition for product {pid}"
        files = list(partition_dir.glob("*.parquet"))
        assert len(files) == 12, f"Expected 12 files for product {pid}, got {len(files)}"
    
    print("Validation checks passed!")


if __name__ == "__main__":
    main()