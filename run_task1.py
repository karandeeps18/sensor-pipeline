import sys
import time
sys.path.insert(0, ".")

from src.visualization import generate_all_graphics

from src.analysis import(
    run_analysis,
    get_base_sensor_ids,
    get_extended_sensor_ids,
    get_uk_non_holiday_mondays,
    get_time_buckets
)

def main():
    mondays = get_uk_non_holiday_mondays(2020)
    buckets = get_time_buckets()
    base_sensors = get_base_sensor_ids()
    
    print(f"UK non holidays modays: {len(mondays)}")
    print(f"10 min time buckets: {len(buckets)}")
    print(f"base sensors: {len(base_sensors)}")
    
    # on base senors 
    start_time = time.time()
    per_day, across_day = run_analysis(base_sensors, output_suffix="")
    elapsed = time.time() - start_time 
    
    print(f"  Per-day rows: {len(per_day):,}")
    print(f"  Across-days rows: {len(across_day):,}")

    # on extended sensors
    extended_sensors = get_extended_sensor_ids()
    print(f"Extended sensors (id % 3 == 0): {len(extended_sensors)}")

    start_time = time.time()
    per_day_ext, across_day_ext = run_analysis(extended_sensors, output_suffix="_extended")
    elapsed = time.time() - start_time 
    print(f"  Per-day rows: {len(per_day_ext):,}")
    print(f"  Across-days rows: {len(across_day_ext):,}")
    
    # generate graphics 
    start_time = time.time()
    generate_all_graphics()
    elapsed = time.time() - start_time

if __name__ == "__main__":
    main()
