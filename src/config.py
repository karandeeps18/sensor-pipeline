"""Constant for iot environment configuration."""
from pathlib import Path

# time range (20200-01-01 to 2020-12-31)
YEAR = 2020

# Sensor configuration
NUM_BASE_SENSORS = 2001  # field_0 to field_2000
NUM_NEW_SENSORS = 20     # field_2001 to field_2020
NUM_PRODUCTS = 10       # product_id 0 to 10

# Data generation
LAMBDA_PER_HOUR = 20  # Poisson rate: avg events per hour per (product, sensor)
MIN_EVENTS_PER_HOUR = 1

# Storage
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "output" / "data"
RESULTS_DIR = BASE_DIR / "output" / "results"

# batch size
DAYS_PER_BATCH = 30