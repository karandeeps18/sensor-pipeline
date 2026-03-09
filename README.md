# IoT Time-Series Data Pipeline

A scalable Python system for generating, processing, and analyzing large-scale time-series sensor data (~60GB).

### Running Tasks

**1. Generate Data (~60GB)**

```bash
python run_generation.py
```

- **Output:** `output/data/product_id={1-10}/batch_*.parquet`
- **Time:** ~30-60 minutes (depending on hardware)
- **Size:** ~60GB partitioned Parquet files

**2. Batch Analysis (Task 1)**

```bash
python run_task1.py
```

- **Output:**
    - Aggregations per Monday, per sensor, per 10-min bucket
    - Aggregations averaged across all Mondays
    - Same with new sensors (2001-2020)
    - Same with new sensors
    - Distribution plots for sensors [10, 100, 1000]

**3. Streaming Simulation (Task 2)**

```bash
python run_task2.py
```

- **Output:** `output/results/snapshots/snapshots.parquet`
- **Contents:** 30-second snapshots of all sensor values (field_x + p_k)

## Project Structure

```
├── src/
│   ├── config.py        # Constants and configuration
│   ├── generator.py     # Data generation (Poisson timestamps, Gaussian values)
│   ├── analysis.py      # Batch analysis (UK Mondays, 10-min aggregations)
│   ├── streaming.py     # Streaming simulation (replay + p_k sensors)
│   └── visualization.py # Graphics generation
├── run_generation.py    # Entry point: generate 60GB data
├── run_task1.py         # Entry point: batch analysis + graphics
├── run_task2.py         # Entry point: streaming simulation
├── setup.sh             # Environment setup
├── requirements.txt     # Dependencies
└── README.md
```

## Design & Architecture

### Storage: Partitioned Parquet with ZSTD

I store simulated time-series data as a **partitioned Parquet dataset** backed by Apache Arrow.

**Why Parquet?**
- Columnar format with excellent compression (ZSTD: ~3-5x)
- Predicate pushdown for efficient filtering
- Native support in Pandas, PyArrow, DuckDB, Spark

**Why partition by `product_id`?**
- Query for one product → read only that folder (~6GB instead of 60GB)
- Natural parallelism boundary
- Avoids thousands of tiny files

**Schema (fixed dtypes for minimal footprint):**

| **Column** | **Type** | **Bytes** |
| ---------- | -------- | --------- |
| epoch_ns   | int64    | 8         |
| product_id | uint8    | 1         |
| sensor_id  | int16    | 2         |
| value      | float32  | 4         |

### Memory Management

The pipeline processes data in bounded chunks to fit in ~16GB RAM:
- **Generation:** One product × one month at a time (~480MB per batch)
- **Analysis:** One product at a time, aggregate, free memory, repeat
- **Streaming:** Sequential replay with dict-based state tracking

### Data Generation

- **Timestamps:** Poisson process with λ=22 events/hour/sensor, uniform within each hour
- **Values:** Gaussian distribution with μ(x,p) = x/10 - 1 + p, σ(x,p) = log₁₀(x+1) + p
- **Validation:** All values enforced finite (no NaN/inf)

### Batch Analysis (Task 1)

1. **Filter:** UK non-holiday Mondays (50 days), 09:30-12:30 UK time
2. **Sensors:** Only where `sensor_id % 3 == 0` (667 base sensors)
3. **Aggregate:** Mean and std in 18 × 10-minute buckets
4. **Extensibility:** Re-run with 20 new sensors (field_2001 to field_2020)

### Streaming Simulation (Task 2)

- **Replay:** First 5 days of historical data, ordered by timestamp
- **p_k sensors:** Jump ±k every 10s (50/50), exponential decay with half-life 5s
- **Snapshots:** Every 30 simulated seconds, capture all sensor values

## Strengths & Weaknesses

### Strengths

- **Simple and readable:** Pure Python/Pandas, easy to understand and modify
- **Memory-bounded:** Processes 60GB on 16GB RAM machine
- **Correct partitioning:** Predicate pushdown on product_id reduces I/O
- **Fixed dtypes:** Minimal memory footprint per row
- **ZSTD compression:** Better compression ratio than Snappy

### Weaknesses

- **Single-threaded:** Does not utilize multiple cores for parallel processing
- **Time filtering in Python:** Parquet predicate pushdown on `epoch_ns` not fully utilized
- **Full scan for Mondays:** Must read all data for a product, then filter by day/time in Pandas
- **No incremental processing:** Re-runs full analysis even for small data additions

## Trade-offs

| Choice                        | Benefit                       | Cost                              |
| ----------------------------- | ----------------------------- | --------------------------------- |
| Partition by product only     | Simple structure, fewer files | Time-based queries scan all files |
| Sequential product processing | Bounded memory, simple code   | Slower than parallel              |
| Pandas for aggregations       | Clear, debuggable             | Not fastest for 60GB scale        |
| Python over Spark/Dask        | No cluster setup, easy to run | Single-node only                  |
|                               |                               |                                   |

## Future Improvements

Given more time, I would have:
1. **Time-based partitioning:** Add `month=YYYY-MM` partition to skip files for time-range queries
2. **Predicate pushdown on epoch_ns:** Filter rows at Arrow level, not Pandas level
3. **Parallel processing:** Use `multiprocessing` to analyze products concurrently
4. **Faster engine:** Replace Pandas aggregations with Polars or DuckDB for 5-10x speedup
5. **Column pruning:** Only read required columns in `dataset.to_table(columns=[...])`
6. **Pre-computed buckets:** Store time bucket as a column during generation
7. **Better monitoring:** Add memory profiling, bytes read/written per stage
