import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds

from src.config import DATA_DIR, RESULTS_DIR
from src.analysis import (
    get_uk_non_holiday_mondays,
    filter_uk_monday_window,
    WINDOW_START,
    WINDOW_END,
    UK_TZ,
)


# Sensors to visualize
TARGET_SENSORS = [10, 100, 1000]


def load_sensor_data_by_product(data_dir: Path, sensor_id: int) -> pd.DataFrame:
    """Load data for a specific sensor across all products."""
    partitioning = ds.partitioning(
        pa.schema([("product_id", pa.uint8())]),
        flavor="hive"
    )
    
    dataset = ds.dataset(data_dir, format="parquet", partitioning=partitioning)
    
    table = dataset.to_table(
        filter=ds.field("sensor_id") == sensor_id
    )
    
    return table.to_pandas()


def plot_sensor_distribution_by_product(sensor_id: int, output_dir: Path):
    """Plot value distribution for a sensor across all product_ids.
    
    Creates a box plot showing the distribution of values for each product_id
    across all non-holiday UK Mondays, 09:30-12:30.
    """
    print(f"  Generating plot for sensor {sensor_id}...")
    
    # Load data for this sensor
    df = load_sensor_data_by_product(DATA_DIR, sensor_id)
    
    if df.empty:
        print(f"    Warning: No data found for sensor {sensor_id}")
        return
    
    # Filter to UK Monday windows
    mondays = get_uk_non_holiday_mondays(2020)
    df = filter_uk_monday_window(df, mondays)
    
    if df.empty:
        print(f"    Warning: No data after filtering for sensor {sensor_id}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by product_id and collect values
    products = sorted(df["product_id"].unique())
    data_by_product = [df[df["product_id"] == p]["value"].values for p in products]
    
    # Box plot
    bp = ax.boxplot(data_by_product, labels=products, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(products)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Labels and title
    ax.set_xlabel("Product ID", fontsize=12)
    ax.set_ylabel("Sensor Value", fontsize=12)
    ax.set_title(f"Sensor {sensor_id} - Value Distribution by Product\n(UK Mondays, 09:30-12:30)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    
    # Save
    output_file = output_dir / f"sensor_{sensor_id}_by_product.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"    Saved: {output_file}")


def plot_sensor_stats_summary(output_dir: Path):
    """Plot summary statistics for target sensors across products."""
    print("  Generating summary plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, sensor_id in enumerate(TARGET_SENSORS):
        ax = axes[idx]
        
        # Load and filter data
        df = load_sensor_data_by_product(DATA_DIR, sensor_id)
        if df.empty:
            continue
            
        mondays = get_uk_non_holiday_mondays(2020)
        df = filter_uk_monday_window(df, mondays)
        
        if df.empty:
            continue
        
        # Compute stats per product
        stats = df.groupby("product_id")["value"].agg(["mean", "std"]).reset_index()
        
        # Bar plot with error bars
        x = stats["product_id"]
        y = stats["mean"]
        yerr = stats["std"]
        
        bars = ax.bar(x, y, yerr=yerr, capsize=3, alpha=0.7, color=plt.cm.tab10(idx))
        ax.set_xlabel("Product ID")
        ax.set_ylabel("Mean Value")
        ax.set_title(f"Sensor {sensor_id}")
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Mean ± Std by Product", fontsize=14)
    plt.tight_layout()
    
    output_file = output_dir / "sensors_summary_by_product.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"    Saved: {output_file}")


def generate_all_graphics():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Individual sensor plots
    for sensor_id in TARGET_SENSORS:
        plot_sensor_distribution_by_product(sensor_id, RESULTS_DIR)
    
    # Summary plot
    plot_sensor_stats_summary(RESULTS_DIR)
    
    print("Graphics generation complete.")